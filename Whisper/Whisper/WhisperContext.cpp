#include "stdafx.h"
#include "WhisperContext.h"
#include "ModelBuffers.h"
#include <optional>
#include "../Utils/Trace/tracing.h"
#include "../D3D/RenderDoc/renderDoc.h"
#include "../ML/testUtils.h"
using namespace DirectCompute;

namespace
{
	// True to measure GPU time of individual shaders which run during the encode step of the algorithm
	constexpr bool profileEncodeShaders = true;
	// True to measure GPU time of individual shaders which run during the decode step of the algorithm
	constexpr bool profileDecodeShaders = true;

	LPCTSTR traceFileNative = LR"(C:\Temp\2remove\Whisper\gpu.bin)";
	LPCTSTR traceFileHybrid = LR"(C:\Temp\2remove\Whisper\hybrid.bin)";

	TensorsArena::sArenaConfigs defaultArenaConfigs()
	{
		TensorsArena::sArenaConfigs res = {};
		return res;
	}
}

WhisperContext::Arenas::Arenas() :
	outer( defaultArenaConfigs() ), layer( defaultArenaConfigs() )
{ }

Tensor WhisperContext::DecoderLayerPool::tensor( eDataType type, const std::array<uint32_t, 4>& ne )
{
	assert( type == eDataType::FP32 );
	return result.tensor( eDataType::FP32, ne, &DirectCompute::defaultNewCapacity );
}

class WhisperContext::ArenaRaii
{
	WhisperContext& context;
	iTensorArena* prevCurrent;

public:

	ArenaRaii( WhisperContext& ctx, iTensorArena& ta ) :
		context( ctx )
	{
		prevCurrent = ctx.currentArena;
		ctx.currentArena = &ta;
	}

	~ArenaRaii()
	{
		context.currentArena->reset();
		context.currentArena = prevCurrent;
	}
};

WhisperContext::WhisperContext( const Whisper::WhisperModel& wm, Whisper::ProfileCollection& pc ) :
	MlContext( pc ),
	gpuModel( wm.tensors )
{
#if BUILD_HYBRID_VERSION
	if( !wm.shared->hybridTensors.layers.empty() )
	{
		hybridContext = std::make_unique<HybridContext>( wm );
		check( hybridContext->create() );
#if SAVE_DEBUG_TRACE
		Tracing::traceCreate( traceFileHybrid );
#endif
	}
	else
#endif
	{
#if SAVE_DEBUG_TRACE
		Tracing::traceCreate( traceFileNative );
#endif
	}
}

#if BUILD_BOTH_VERSIONS
namespace
{
	thread_local WhisperContext* ts_context = nullptr;
	const ModelBuffers& getGlobalModel()
	{
		return gpuModel;
	}
}

/*
WhisperContext::WhisperContext() :
	gpuModel( getGlobalModel() ),
{
	if( nullptr != ts_context )
		throw HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );
	ts_context = this;
}*/

WhisperContext::~WhisperContext()
{
	Tracing::traceClose();
	if( ts_context == nullptr )
		return;
	assert( ts_context == this );
	ts_context = nullptr;
}

WhisperContext& WhisperContext::current()
{
	WhisperContext* c = ts_context;
	if( nullptr == c )
		throw OLE_E_BLANK;
	return *c;
}
#else
WhisperContext& WhisperContext::current()
{
	throw E_NOTIMPL;
}
#endif

Tensor WhisperContext::createTensor( eDataType type, const std::array<uint32_t, 4>& ne )
{
	// return MlContext::createTensor( type, ne );

	iTensorArena* const ca = currentArena;
	if( nullptr != ca )
		return ca->tensor( type, ne );
	else
		return MlContext::createTensor( type, ne );
}

void WhisperContext::fmaRepeat( Tensor& cur, const TensorPair& that )
{
	MlContext::fmaRepeat( cur, that.w, that.b );
}

Tensor WhisperContext::convolutionAndGelu( const Tensor& mel, uint32_t n_ctx )
{
	const EncoderBuffers& model = gpuModel.enc;
	Tensor cur = conv_1d_1s( model.conv1.w, mel );
	Tracing::tensor( "enc.conv1", cur );
	addRepeatGelu( cur, model.conv1.b );
	Tracing::tensor( "enc.temp1", cur );

	cur = conv_1d_2s( model.conv2.w, cur );
	addRepeatGelu( cur, model.conv2.b );

	const Tensor& posEmbed = model.positionalEmbedding;
	const uint32_t peStride = posEmbed.ne[ 0 ];
	constexpr uint32_t peOffset = 0;

	Tensor e_pe = view2d( posEmbed, posEmbed.ne[ 0 ], n_ctx, peStride, peOffset );
	cur = add( e_pe, transpose( cur ) );
	return cur;
}

Tensor WhisperContext::encodeLayer( const Tensor& source, size_t index, uint32_t n_state, uint32_t n_head, uint32_t n_ctx )
{
	auto prof = profiler.block( eProfilerBlock::EncodeLayer );
	ArenaRaii arenaRaii{ *this, arenas.layer };

	const LayerEncoder& layer = gpuModel.enc.layers[ index ];
	// norm
	Tensor cur = norm( source );
	if( 0 == index )
		Tracing::tensor( "enc-norm", cur );
	fmaRepeat( cur, layer.attnLn0 );

	// self-attention
	Tensor Qcur;
	Tensor reshaped;
	if( gpuInfo().useReshapedMatMul() )
	{
		const uint16_t tag = profiler.setNextTag( "enc.layer.1" );
		reshaped = reshapePanels( cur );
		profiler.setNextTag( tag );
		Qcur = mulMatTiledEx( layer.attnQuery.w, reshaped );
	}
	else
	{
		profiler.setNextTag( "enc.layer.1" );
		Qcur = mulMat( layer.attnQuery.w, cur );
	}

	if( 0 == index )
		Tracing::tensor( "enc-Qcur", Qcur );
	addRepeat( Qcur, layer.attnQuery.b );

	// note: no bias for Key
	Tensor Kcur;
	if( gpuInfo().useReshapedMatMul() )
	{
		// Already reshaped by the previous `if`
		profiler.setNextTag( "enc.layer.2" );
		Kcur = mulMatTiledEx( layer.attnKey, reshaped );
	}
	else
	{
		profiler.setNextTag( "enc.layer.2" );
		Kcur = mulMat( layer.attnKey, cur );
	}

	if( 0 == index )
		Tracing::tensor( "enc-Kcur", Kcur );

	Tensor Vcur;
	if( gpuInfo().useReshapedMatMul() )
	{
		// Already reshaped by the previous `if`
		profiler.setNextTag( "enc.layer.3" );
		Vcur = mulMatTiledEx( layer.attnValue.w, reshaped );
	}
	else
	{
		profiler.setNextTag( "enc.layer.3" );
		Vcur = mulMat( layer.attnValue.w, cur );
	}

	if( 0 == index )
		Tracing::tensor( "enc-Vcur", Vcur );
	addRepeat( Vcur, layer.attnValue.b );

	// ------
	Tensor Q = permute( copy( Qcur, eDataType::FP16, { n_state / n_head, n_head, n_ctx } ), 0, 2, 1, 3 );
	Tensor K = permute( copy( Kcur, eDataType::FP16, { n_state / n_head, n_head, n_ctx } ), 0, 2, 1, 3 );
	Tensor V = copy( permute( Vcur.reshape3d( n_state / n_head, n_head, n_ctx ), 1, 2, 0, 3 ), eDataType::FP16, { n_ctx, n_state / n_head, n_head } );
	Tensor KQV = flashAttention( Q, K, V, false );
	if( 0 == index )
		Tracing::tensor( "enc-KQV", KQV );
	Tensor KQV_merged = permute( KQV, 0, 2, 1, 3 );
	copyInPlace( cur, KQV_merged, eDataType::FP32, { n_state, n_ctx } );

	// projection
	if( gpuInfo().useReshapedMatMul() )
	{
		const uint16_t tag = profiler.setNextTag( "enc.layer.4" );
		cur = reshapePanels( cur );
		profiler.setNextTag( tag );
		cur = mulMatTiledEx( layer.attnLn1.w, cur );
	}
	else
	{
		profiler.setNextTag( "enc.layer.4" );
		cur = mulMat( layer.attnLn1.w, cur );
	}

	// add the input
	addRepeatEx( cur, layer.attnLn1.b, source );

	// feed-forward network
	Tensor inpFF = cur;

	cur = norm( inpFF );
	fmaRepeat( cur, layer.mlpLn );

	// fully connected
	if( gpuInfo().useReshapedMatMul() )
	{
		const uint16_t tag = profiler.setNextTag( "enc.layer.5" );
		cur = reshapePanels( cur );
		profiler.setNextTag( tag );
		cur = mulMatTiledEx( layer.mlp0.w, cur );
	}
	else
	{
		profiler.setNextTag( "enc.layer.5" );
		cur = mulMat( layer.mlp0.w, cur );
	}
	addRepeatGelu( cur, layer.mlp0.b );

	// projection
	if( gpuInfo().useReshapedMatMul() )
	{
		const uint16_t tag = profiler.setNextTag( "enc.layer.6" );
		cur = reshapePanels( cur );
		profiler.setNextTag( tag );
		cur = mulMatTiledEx( layer.mlp1.w, cur );
	}
	else
	{
		profiler.setNextTag( "enc.layer.6" );
		cur = mulMat( layer.mlp1.w, cur );
	}

	// output from this layer
	addRepeatEx( cur, layer.mlp1.b, inpFF );
	return cur;
}

void WhisperContext::createKeyValueBuffers( const sEncodeParams& encParams )
{
	{
		const uint32_t n_audio_ctx = encParams.n_audio_ctx;
		const uint32_t n_mem = encParams.n_text_layer * encParams.n_audio_ctx;
		const uint32_t n_elements = encParams.n_text_state * n_mem;
		kvCross.resize( n_elements );
	}

#if BUILD_HYBRID_VERSION
	if( !hybridContext )
#endif
	{
		const uint32_t n_mem = encParams.n_text_layer * encParams.n_text_ctx;
		const uint32_t n_elements = encParams.n_text_state * n_mem;
		kv.resize( n_elements );
	}
}

Tensor WhisperContext::encode( Whisper::iSpectrogram& spectrogram, const sEncodeParams& encParams )
{
	auto prof = profiler.block( eProfilerBlock::Encode );
	CaptureRaii renderdocCapture;
	profiler.profileShaders = profileEncodeShaders;

	createKeyValueBuffers( encParams );
	// Upload the source
	check( melInput.create( spectrogram, encParams ) );
	Tracing::tensor( "enc.input", melInput );

	ArenaRaii arenaRaii{ *this, arenas.outer };

	// Initial few steps
	Tensor cur = convolutionAndGelu( melInput, encParams.n_ctx );

	// Process all these layers
	{
		const size_t layersCount = encParams.layersCount;
		for( size_t i = 0; i < layersCount; i++ )
		{
			Tracing::tensor( { "enc.layer[ %i ].in", i }, cur );
			cur = encodeLayer( cur, i, encParams.n_state, encParams.n_head, encParams.n_ctx );
		}
	}
	Tracing::tensor( "enc.layers", cur );

	// A few last steps
	{
		cur = norm( cur );
		// cur = ln_f_g*cur + ln_f_b
		fmaRepeat( cur, gpuModel.enc.lnPost );
	}

	// pre-compute cross-attention buffers
	{
		Tensor reshaped;
		if( gpuInfo().useReshapedMatMul() )
		{
			if( cur.ne[ 1 ] != 1 )
			{
				profiler.setNextTag( "enc.cross" );
				reshaped = reshapePanels( cur );
			}
			else
				reshaped = cur;
		}

		const size_t layersCount = encParams.n_text_layer;
		const uint32_t stride = encParams.n_state * encParams.n_ctx;
		const float finalScaling = computeScaling( (int)encParams.n_state, (int)encParams.n_head );
		for( size_t i = 0; i < layersCount; i++ )
		{
			const LayerDecoder& layer = gpuModel.dec.layers[ i ];
			Tensor Kcross, Vcross;
			if( gpuInfo().useReshapedMatMul() )
				Kcross = mulMatEx( layer.crossAttnKey, reshaped, "enc.cross.1" );
			else
			{
				profiler.setNextTag( "enc.cross.1" );
				Kcross = mulMat( layer.crossAttnKey, cur );
			}
			scale( Kcross, finalScaling );

			if( gpuInfo().useReshapedMatMul() )
				Vcross = mulMatEx( layer.crossAttnValue.w, reshaped, "enc.cross.2" );
			else
			{
				profiler.setNextTag( "enc.cross.2" );
				Vcross = mulMat( layer.crossAttnValue.w, cur );
			}
			addRepeat( Vcross, layer.crossAttnValue.b );

			Tensor k = kvCross.keys.view( stride, stride * (uint32_t)i );
			copyImpl( Kcross, k, Kcross.getType() == eDataType::FP32 );

			Tensor v = kvCross.values.view( stride, stride * (uint32_t)i );
			copyImpl( Vcross, v, Vcross.getType() == eDataType::FP32 );
		}
	}

#if BUILD_HYBRID_VERSION
	if( hybridContext )
	{
		// When running hybrid model, download cross-attention buffers from VRAM to system RAM
		check( hybridContext->downloadKeyValues( kvCross ) );
	}
#endif
	return cur;
}

struct WhisperContext::sLayerDecParams
{
	uint32_t n_state, n_head, N;
	uint32_t n_ctx, n_past, M;
};

Tensor WhisperContext::decodeLayer( const Tensor& inpL, size_t il, const sLayerDecParams& ldp )
{
	auto prof = profiler.block( eProfilerBlock::DecodeLayer );
	const auto& layer = gpuModel.dec.layers[ il ];
	std::optional<ArenaRaii> arenaRaii{ std::in_place, *this, arenas.layer };
	if( 0 == il ) Tracing::tensor( "dec-inpL", inpL );

	// norm
	Tensor cur = norm( inpL );
	fmaRepeat( cur, layer.attnLn0 );
	if( 0 == il ) Tracing::tensor( "dec-norm", cur );

	// self-attention
	{
		profiler.setNextTag( "dec.layer.1" );
		Tensor Qcur = mulMat( layer.attnQuery.w, cur );
		if( 0 == il ) Tracing::tensor( "dec-Qcur-0", Qcur );
		const float scaling = computeScaling( (int)ldp.n_state, (int)ldp.n_head );
		addRepeatScale( Qcur, layer.attnQuery.b, scaling );
		if( 0 == il ) Tracing::tensor( "dec-Qcur-1", Qcur );

		// note: no bias for Key
		profiler.setNextTag( "dec.layer.2" );
		Tensor Kcur = mulMat( layer.attnKey, cur );
		scale( Kcur, scaling );
		if( 0 == il ) Tracing::tensor( "dec-Kcur", Kcur );

		profiler.setNextTag( "dec.layer.3" );
		Tensor Vcur = mulMat( layer.attnValue.w, cur );
		addRepeat( Vcur, layer.attnValue.b );
		if( 0 == il ) Tracing::tensor( "dec-Vcur", Vcur );

		// store key and value to memory
		{
			const uint32_t len = ldp.N * ldp.n_state;
			const uint32_t off = ldp.n_state * ( (uint32_t)il * ldp.n_ctx + ldp.n_past );
			Tensor k = kv.keys.view( len, off );
			Tensor v = kv.values.view( len, off );
			copyImpl( Kcur, k, true );
			copyImpl( Vcur, v, true );
		}

		// ------
		Tensor Q = permute( copy( Qcur, eDataType::FP32, { ldp.n_state / ldp.n_head, ldp.n_head, ldp.N } ), 0, 2, 1, 3 );
		Tensor K = permute( kv.keys.view( ( ldp.n_past + ldp.N ) * ldp.n_state, (uint32_t)il * ldp.n_ctx * ldp.n_state )
			.reshape3d( ldp.n_state / ldp.n_head, ldp.n_head, ldp.n_past + ldp.N ),
			0, 2, 1, 3 );
		profiler.setNextTag( "dec.layer.4" );
		Tensor KQ = mulMat( K, Q );
		if( 0 == il ) Tracing::tensor( "dec-KQ-0", KQ );
		diagMaskInf( KQ, ldp.n_past );
		if( 0 == il ) Tracing::tensor( "dec-KQ-1", KQ );
		profiler.setNextTag( "decLayer.1" );
		softMax( KQ );
		if( 0 == il ) Tracing::tensor( "dec-KQ-2", KQ );

		Tensor V_trans = permute(
			kv.values
			.view( ( ldp.n_past + ldp.N ) * ldp.n_state, (uint32_t)il * ldp.n_ctx * ldp.n_state )
			.reshape3d( ldp.n_state / ldp.n_head, ldp.n_head, ldp.n_past + ldp.N ),
			1, 2, 0, 3 );

		profiler.setNextTag( "dec.layer.5" );
		Tensor KQV = mulMat( V_trans, KQ );
		if( 0 == il ) Tracing::tensor( "dec-KQV", KQV );

		Tensor KQV_merged = permute( KQV, 0, 2, 1, 3 );
		copyInPlace( cur, KQV_merged, eDataType::FP32, { ldp.n_state, ldp.N } );
	}

	{
		profiler.setNextTag( "dec.layer.6" );
		cur = mulMat( layer.attnLn1.w, cur );
	}
	// add the input
	addRepeatEx( cur, layer.attnLn1.b, inpL );
	Tensor inpCA = cur;

	// norm
	{
		cur = norm( inpCA );
		fmaRepeat( cur, layer.crossAttnLn0 );
	}

	// cross-attention
	{
		profiler.setNextTag( "dec.layer.7" );
		Tensor Qcur = mulMat( layer.crossAttnQuery.w, cur );
		addRepeatScale( Qcur, layer.crossAttnQuery.b, computeScaling( (int)ldp.n_state, (int)ldp.n_head ) );

		// Kcross is already scaled
		const uint32_t len = ldp.M * ldp.n_state;
		const uint32_t off = (uint32_t)il * len;
		Tensor Kcross = kvCross.keys.view( len, off ).reshape3d( ldp.n_state / ldp.n_head, ldp.n_head, ldp.M );
		Tensor Vcross = kvCross.values.view( len, off ).reshape3d( ldp.n_state / ldp.n_head, ldp.n_head, ldp.M );

		// ------
		Tensor Q = permute( copy( Qcur, eDataType::FP32, { ldp.n_state / ldp.n_head, ldp.n_head, ldp.N } ), 0, 2, 1, 3 );
		Tensor K = permute( Kcross, 0, 2, 1, 3 );
		profiler.setNextTag( "dec.layer.8" );
		Tensor KQ = mulMat( K, Q );
		profiler.setNextTag( "decLayer.2" );
		softMax( KQ );
		Tensor V_trans = permute( Vcross, 1, 2, 0, 3 );
		profiler.setNextTag( "dec.layer.9" );
		Tensor KQV = mulMat( V_trans, KQ );
		if( 0 == il ) Tracing::tensor( "dec-KQV", KQV );
		Tensor KQV_merged = permute( KQV, 0, 2, 1, 3 );

		copyInPlace( cur, KQV_merged, eDataType::FP32, { ldp.n_state, ldp.N } );
	}

	// projection
	{
		profiler.setNextTag( "dec.layer.10" );
		cur = mulMat( layer.crossAttnLn1.w, cur );
	}
	// add the input
	addRepeatEx( cur, layer.crossAttnLn1.b, inpCA );
	Tensor inpFF = cur;

	// feed-forward network
	{
		// norm
		cur = norm( inpFF );
		fmaRepeat( cur, layer.mlpLn );

		if( gpuInfo().useReshapedMatMul() )
			cur = mulMatEx( layer.mlp0.w, cur, "dec.layer.11" );
		else
		{
			profiler.setNextTag( "dec.layer.11" );
			cur = mulMat( layer.mlp0.w, cur );
		}

		addRepeatGelu( cur, layer.mlp0.b );

		// projection
		if( gpuInfo().useReshapedMatMul() )
		{
			if( cur.ne[ 1 ] != 1 )
			{
				const uint16_t tag = profiler.setNextTag( "dec.layer.12" );
				cur = reshapePanels( cur );

				// The mulMatTiledEx() line creates a layer output tensor. We have a special pool for such tensors so they survive the destruction of the arena.
				arenaRaii.emplace( *this, decPool );
				profiler.setNextTag( tag );
				cur = mulMatTiledEx( layer.mlp1.w, cur );
			}
			else
			{
				// The mulMatByRowTiledEx() line creates a layer output tensor. We have a special pool for such tensors so they survive the destruction of the arena.
				arenaRaii.emplace( *this, decPool );
				profiler.setNextTag( "dec.layer.12" );
				cur = mulMatByRowTiledEx( layer.mlp1.w, cur );
			}
		}
		else
		{
			// The mulMat() line creates a layer output tensor. We have a special pool for such tensors so they survive the destruction of the arena.
			arenaRaii.emplace( *this, decPool );
			profiler.setNextTag( "dec.layer.12" );
			cur = mulMat( layer.mlp1.w, cur );
		}
	}
	// output from this layer
	addRepeatEx( cur, layer.mlp1.b, inpFF );
	return cur;
}

void WhisperContext::decode( const int* tokens, const int n_tokens, const sDecodeParams& decParams, std::vector<float>& probs, int threads )
{
	auto cppp = profiler.cpuBlock( Whisper::eCpuBlock::DecodeStep );

#if BUILD_HYBRID_VERSION
	if( hybridContext )
	{
		HybridContext::sDecParams sdp;
		sdp.n_threads = threads;
		sdp.M = decParams.M;
		check( hybridContext->decode( tokens, n_tokens, decParams.n_past, sdp, probs ) );
		return;
	}
#endif

	auto prof = profiler.block( eProfilerBlock::DecodeStep );
	CaptureRaii renderdocCapture;
	profiler.profileShaders = profileDecodeShaders;
	ArenaRaii arenaRaii{ *this, arenas.outer };

	assert( n_tokens > 0 );
	const uint32_t N = (uint32_t)n_tokens;
	decoderInput.resize( N );

	Tensor embd = decoderInput.embedding( tokens );
	Tensor cur = addRows( gpuModel.dec.tokenEmbedding, gpuModel.dec.positionalEmbedding, embd, decParams.n_past );
	Tracing::tensor( "dec-rows", cur );

	{
		sLayerDecParams ldp;
		ldp.n_state = decParams.n_state;
		ldp.n_head = decParams.n_head;
		ldp.N = N;
		ldp.n_ctx = decParams.n_ctx;
		ldp.n_past = decParams.n_past;
		ldp.M = decParams.M;
#if 1
		for( size_t i = 0; i < decParams.n_text_layer; i++ )
			cur = decodeLayer( cur, i, ldp );
#else
		dbgDecodeTest = decodeLayer( cur, 0, ldp );
		return;
#endif
	}

	// norm
	cur = norm( cur );
	fmaRepeat( cur, gpuModel.dec.ln );

	profiler.setNextTag( "dec.logits" );
	cur = mulMat( gpuModel.dec.tokenEmbedding, cur );

	// logits -> probs
	profiler.setNextTag( "dec.probs" );
	softMax( cur );

	decoderOutput.copyFromVram( cur );
	assert( decoderOutput.size() == N * decParams.n_vocab );

	decoderOutput.copyToVector( probs );
	Tracing::vector( "probs", probs );
}

__m128i WhisperContext::Arenas::getMemoryUse() const
{
	__m128i res = outer.getMemoryUse();
	res = _mm_add_epi64( res, layer.getMemoryUse() );
	return res;
}

__m128i WhisperContext::DecoderLayerPool::getMemoryUse() const
{
	size_t cb = result.getCapacity() * 4;
	__m128i res = _mm_setzero_si128();
	return _mm_insert_epi64( res, (int64_t)cb, 1 );
}

__m128i WhisperContext::getMemoryUse() const
{
	__m128i res = MlContext::getMemoryUse();
	res = _mm_add_epi64( res, arenas.getMemoryUse() );
	res = _mm_add_epi64( res, decPool.getMemoryUse() );
	res = _mm_add_epi64( res, melInput.getMemoryUse() );
	res = _mm_add_epi64( res, kv.getMemoryUse() );
	res = _mm_add_epi64( res, kvCross.getMemoryUse() );
	res = _mm_add_epi64( res, decoderInput.getMemoryUse() );
	res = _mm_add_epi64( res, decoderOutput.getMemoryUse() );
	return res;
}

HRESULT WhisperContext::clearState()
{
	// CHECK( kv.zeroMemory( cb ) );
	// CHECK( kvCross.zeroMemory( cb ) );
	// The above code doesn't work for some reason.
	// Ideally need to debug, but destroying and re-creating these two buffers is not a huge deal. Unlike the buffers in the pools, only a few megabytes of VRAM.
	kv.clear();
	kvCross.clear();

	CHECK( arenas.outer.zeroMemory() );
	CHECK( arenas.layer.zeroMemory() );
	CHECK( decPool.zeroMemory() );
	CHECK( decoderInput.zeroMemory() );
	decoderOutput.clear();
	return S_OK;
}