#include "stdafx.h"
#include <immintrin.h>
#include <optional>
#include "HybridContext.h"
#include "../Utils/Trace/tracing.h"

#if BUILD_HYBRID_VERSION
#ifndef __AVX__
#error Hybrid version requires AVX build, and ideally AVX2 CPU
#endif // !__AVX__

namespace
{
	int threadsCount( int t )
	{
#ifdef NDEBUG
		if( t == 0 )
		{
			SYSTEM_INFO si;
			GetSystemInfo( &si );
			return (int)si.dwNumberOfProcessors;
		}
		if( t <= 1 )
			return 1;
		return t;
#else
		return 1;
#endif
	}

	constexpr size_t MB = 1u << 20;
}

HybridContext::HybridContext( const Whisper::WhisperModel& wm ) :
	ml( threadsCount( 0 ) ),
	model( wm.shared->hybridTensors ),
	whisperModel( wm )
{ }

namespace
{
	enum struct eModelType : uint8_t
	{
		Tiny = 0,
		Base = 1,
		Small = 2,
		Medium = 3,
		Large = 4,
	};

	static HRESULT detectModelType( const Whisper::sModelParams& modelParams, eModelType& mt )
	{
		switch( modelParams.n_audio_layer )
		{
		case 4:
			mt = eModelType::Tiny;
			return S_OK;
		case 6:
			mt = eModelType::Base;
			return S_OK;
		case 12:
			mt = eModelType::Small;
			return S_OK;
		case 24:
			mt = eModelType::Medium;
			return S_OK;
		case 32:
			mt = eModelType::Large;
			return S_OK;
		}
		logError( u8"Unrecognized model" );
		return E_INVALIDARG;
	}

	struct alignas( 2 ) RamMB
	{
		uint8_t dec, decLayer;
		constexpr RamMB( uint8_t d, uint8_t dl ) : dec( d ), decLayer( dl ) { }

		__m128i loadBytes() const
		{
			__m128i v = _mm_loadu_si16( this );
			// Upcast bytes to int64_t. That instruction can load directly from memory, too bad VC++ optimized doesn't care
			v = _mm_cvtepu8_epi64( v );
			// Scale from megabytes into bytes, the multiplier is obviously 2^20
			v = _mm_slli_epi64( v, 20 );
			return v;
		}
	};

	// The magic numbers are from MEM_REQ_DECODE and MEM_REQ_DECODE_LAYER red/black maps in the reference version,
	// near the top of whisper.cpp source file
	static const std::array<RamMB, 5> s_memRequirements =
	{
		RamMB{ 200, 32 },	// Tiny
		RamMB{ 202, 44 },	// Base
		RamMB{ 204, 64 },	// Small
		RamMB{ 206, 84 },	// Medium
		RamMB{ 208, 110 },	// Large
	};
}

HRESULT HybridContext::create()
{
	// Allocate buffers for compute
	// We know they're large, so bypassing the heap
	eModelType modelType;
	CHECK( detectModelType( whisperModel.parameters, modelType ) );

	const __m128i bytes = s_memRequirements.at( (uint8_t)modelType ).loadBytes();
	CHECK( allocCompute.create( _mm_cvtsi128_si64( bytes ) ) );
	CHECK( allocComputeLayer.create( _mm_extract_epi64( bytes, 1 ) ) );

	// Create staging buffers to download output from encoder stage,
	// in the reference version they're named memory_cross_k / memory_cross_v
	CHECK( kvCross.create( whisperModel.parameters ) );

	// Create RAM buffers for memory_k / memory_v
	CHECK( kv.create( whisperModel.parameters ) );

	return S_OK;
}

class HybridContext::SetAllocatorRaii
{
	HybridContext& context;
	CpuCompute::iMemoryAllocator* prevAlloc;
	CpuCompute::iArenaAllocator* newAlloc;
public:

	SetAllocatorRaii( HybridContext* owner, CpuCompute::iArenaAllocator& a ) :
		context( *owner )
	{
		prevAlloc = context.ml.setAllocator( &a );
		newAlloc = &a;
	}
	~SetAllocatorRaii()
	{
		context.ml.setAllocator( prevAlloc );
		newAlloc->resetArena();
	}
};

HRESULT HybridContext::decode( const int* tokens, const int n_tokens, const int n_past, const sDecParams& dp, std::vector<float>& probs )
{
	CHECK( ml.setThreadsCount( dp.n_threads ) );

	// whisper_decode
	const auto& hparams = whisperModel.parameters;
	const uint32_t n_vocab = hparams.n_vocab;

	const uint32_t n_ctx = hparams.n_text_ctx;
	const uint32_t n_state = hparams.n_text_state;
	const uint32_t n_head = hparams.n_text_head;
	const uint32_t n_layer = hparams.n_text_layer;

	const uint32_t N = n_tokens;
	const uint32_t M = dp.M;

	SetAllocatorRaii ac{ this, allocCompute };
	using namespace CpuCompute;
	Tensor cur = ml.addRows( model.tokenEmbedding, model.positionalEmbedding, tokens, n_tokens, n_past );
	Tracing::tensor( "dec-rows", cur );

	Tensor inpL = cur;
	auto kvCross = this->kvCross.map();

	for( uint32_t il = 0; il < n_layer; il++ )
	{
		if( 0 == il ) Tracing::tensor( "dec-inpL", inpL );
		const auto& layer = model.layers[ il ];
		SetAllocatorRaii acLayer{ this, allocComputeLayer };

		// norm
		Tensor cur = ml.norm( inpL );
		ml.fmaRepeat( cur, layer.attnLn0 );
		if( 0 == il ) Tracing::tensor( "dec-norm", cur );

		// self-attention
		{
			Tensor Qcur = ml.mulMat( layer.attnQuery.w, cur );
			if( 0 == il ) Tracing::tensor( "dec-Qcur-0", Qcur );
			const float scaling = computeScaling( (int)n_state, (int)n_head );
			ml.addRepeatScale( Qcur, layer.attnQuery.b, scaling );
			if( 0 == il ) Tracing::tensor( "dec-Qcur-1", Qcur );

			// note: no bias for Key
			Tensor Kcur = ml.mulMat( layer.attnKey, cur );
			ml.scale( Kcur, scaling );
			if( 0 == il ) Tracing::tensor( "dec-Kcur", Kcur );

			Tensor Vcur = ml.mulMat( layer.attnValue.w, cur );
			ml.addRepeat( Vcur, layer.attnValue.b );
			if( 0 == il ) Tracing::tensor( "dec-Vcur", Vcur );

			// store key and value to memory
			{
				const uint32_t len = N * n_state;
				const uint32_t off = n_state * ( (uint32_t)il * n_ctx + n_past );
				Tensor k = kv.keysView( len, off );
				Tensor v = kv.valuesView( len, off );

				CHECK( ml.copyImpl( k, Kcur ) );
				CHECK( ml.copyImpl( v, Vcur ) );
			}

			// ------
			Tensor Q = ml.permute( ml.copy( Qcur, eDataType::FP32, { n_state / n_head, n_head, N } ), 0, 2, 1, 3 );
			Tensor K = ml.permute( kv.keysView( ( n_past + N ) * n_state, (uint32_t)il * n_ctx * n_state )
				.reshape3d( n_state / n_head, n_head, n_past + N ),
				0, 2, 1, 3 );
			Tensor KQ = ml.mulMat( K, Q );
			if( 0 == il ) Tracing::tensor( "dec-KQ-0", KQ );
			ml.diagMaskInf( KQ, n_past );
			if( 0 == il ) Tracing::tensor( "dec-KQ-1", KQ );
			ml.softMax( KQ );
			if( 0 == il ) Tracing::tensor( "dec-KQ-2", KQ );

			Tensor V_trans = ml.permute(
				kv.valuesView( ( n_past + N ) * n_state, (uint32_t)il * n_ctx * n_state )
				.reshape3d( n_state / n_head, n_head, n_past + N ),
				1, 2, 0, 3 );

			Tensor KQV = ml.mulMat( V_trans, KQ );
			if( 0 == il ) Tracing::tensor( "dec-KQV", KQV );

			Tensor KQV_merged = ml.permute( KQV, 0, 2, 1, 3 );
			ml.copyInPlace( cur, KQV_merged, eDataType::FP32, { n_state, N } );
		}

		{
			cur = ml.mulMat( layer.attnLn1.w, cur );
			ml.addRepeat( cur, layer.attnLn1.b );
		}

		// add the input
		Tensor inpCA = ml.add( cur, inpL );

		// norm
		{
			cur = ml.norm( inpCA );
			ml.fmaRepeat( cur, layer.crossAttnLn0 );
		}

		// cross-attention
		{
			Tensor Qcur = ml.mulMat( layer.crossAttnQuery.w, cur );
			ml.addRepeatScale( Qcur, layer.crossAttnQuery.b, computeScaling( (int)n_state, (int)n_head ) );

			// Kcross is already scaled
			const uint32_t len = M * n_state;
			const uint32_t off = (uint32_t)il * len;
			Tensor Kcross = kvCross.keysView( len, off ).reshape3d( n_state / n_head, n_head, M );
			Tensor Vcross = kvCross.valuesView( len, off ).reshape3d( n_state / n_head, n_head, M );

			// ------
			Tensor Q = ml.permute( ml.copy( Qcur, eDataType::FP32, { n_state / n_head, n_head, N } ), 0, 2, 1, 3 );
			Tensor K = ml.permute( Kcross, 0, 2, 1, 3 );
			Tensor KQ = ml.mulMat( K, Q );
			ml.softMax( KQ );
			Tensor V_trans = ml.permute( Vcross, 1, 2, 0, 3 );
			Tensor KQV = ml.mulMat( V_trans, KQ );
			if( 0 == il ) Tracing::tensor( "dec-KQV", KQV );
			Tensor KQV_merged = ml.permute( KQV, 0, 2, 1, 3 );

			ml.copyInPlace( cur, KQV_merged, eDataType::FP32, { n_state, N } );
		}

		// projection
		{
			cur = ml.mulMat( layer.crossAttnLn1.w, cur );
			ml.addRepeat( cur, layer.crossAttnLn1.b );
		}
		// add the input
		ml.addInPlace( cur, inpCA );
		Tensor inpFF = cur;

		// feed-forward network
		{
			// norm
			cur = ml.norm( inpFF );
			ml.fmaRepeat( cur, layer.mlpLn );

			cur = ml.mulMat( layer.mlp0.w, cur );
			ml.addRepeatGelu( cur, layer.mlp0.b );

			// The mulMat() below creates a tensor for the output of this layer.
			// We have a special memory storage for these tensors, that's how they survive resets of per-layer arenas
			allocLayerOutput.resetArena();
			ml.setAllocator( &allocLayerOutput );

			// projection
			cur = ml.mulMat( layer.mlp1.w, cur );
			ml.addRepeat( cur, layer.mlp1.b );
		}

		// output from this layer
		ml.addInPlace( cur, inpFF );
		inpL = cur;
	}

	// norm
	cur = ml.norm( inpL );
	ml.fmaRepeat( cur, model.ln );

	cur = ml.mulMat( model.tokenEmbedding, cur );

	// logits -> probs
	ml.softMax( cur );

	const float* rsi = cur.fp32();
	probs.assign( rsi, rsi + cur.countElements() );
	Tracing::vector( "probs", probs );
	return S_OK;
}

void* HybridContext::AllocSingle::allocate( size_t cb, size_t align )
{
	if( !allocated )
	{
		allocated = true;
		if( cb <= capacity )
		{
			CpuCompute::dbgMarkUninitializedMemory( buffer.pointer(), capacity );
			return buffer.pointer();
		}
		else
		{
			HRESULT hr = buffer.allocate( cb );
			if( SUCCEEDED( hr ) )
			{
				capacity = cb;
				CpuCompute::dbgMarkUninitializedMemory( buffer.pointer(), capacity );
				return buffer.pointer();
			}
			logErrorHr( hr, u8"HybridContext.AllocSingle.allocate" );
			throw hr;
		}
	}
	else
	{
		logError( u8"HybridContext.AllocSingle only supports 1 tensor" );
		throw E_UNEXPECTED;
	}
}

void HybridContext::AllocSingle::resetArena()
{
	allocated = false;
	if( capacity > 0 )
		CpuCompute::dbgMarkFreedMemory( buffer.pointer(), capacity );
}
#endif