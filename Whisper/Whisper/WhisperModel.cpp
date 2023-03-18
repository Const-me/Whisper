#include "stdafx.h"
#include "WhisperModel.h"
#include "loaderUtils.h"
#include "../D3D/createBuffer.h"
#include <atlcoll.h>
#include <atlstr.h>
#include "../Utils/GpuProfilerSimple.h"
#include "../Utils/CpuProfiler.h"
#include "../CPU/HybridLoader.h"
#include "../ML/Reshaper.h"
using namespace Whisper;
using namespace DirectCompute;

namespace
{
	struct ParamsAndMelHeader
	{
		sModelParams mp;
		uint32_t n_mel = 0, n_fft = 0;
	};

	enum struct ePostProcessing : uint8_t
	{
		None = 0,
		MakePanels = 1
	};

	struct PendingTensor
	{
		DirectCompute::Tensor* dest = nullptr;
		ePostProcessing postProcessing = ePostProcessing::None;

		PendingTensor() = default;
		PendingTensor( const PendingTensor& ) = default;
		PendingTensor( DirectCompute::Tensor& tensor, ePostProcessing pp = ePostProcessing::None ) :
			dest( &tensor ), postProcessing( pp ) { }

		// If you wonder why not reshape them after all tensors are loaded, doing that on the fly is faster because CPU and GPU work in parallel
		// In the current version, CPU reads data for a next tensor, while in the meantime GPU reshapes a previously loaded tensor.
		HRESULT postProcess( Reshaper& rs, eDataType dt )
		{
			switch( postProcessing )
			{
			case ePostProcessing::None:
				return S_OK;
			case ePostProcessing::MakePanels:
				if( gpuInfo().useReshapedMatMul() )
				{
					// GpuInfo structure says we should use that new method
					return rs.makePanels( *dest, dt );
				}
				else
				{
					// The feature ain't enabled on the current user's GPU
					return S_OK;
				}
			default:
				return E_UNEXPECTED;
			}
		}
	};

	void populateEncodeTensorsMap( CAtlMap<CStringA, PendingTensor>& map, int layersEnc, DirectCompute::ModelBuffers& tensors )
	{
		tensors.enc.layers.resize( layersEnc );

		CStringA tempString;
		// Encoder tensors
		auto& enc = tensors.enc;

		map[ "encoder.positional_embedding" ] = enc.positionalEmbedding;
		map[ "encoder.conv1.weight" ] = enc.conv1.w;
		map[ "encoder.conv1.bias" ] = enc.conv1.b;

		map[ "encoder.conv2.weight" ] = enc.conv2.w;
		map[ "encoder.conv2.bias" ] = enc.conv2.b;

		map[ "encoder.ln_post.weight" ] = enc.lnPost.w;
		map[ "encoder.ln_post.bias" ] = enc.lnPost.b;

		auto add = [ & ]( const char* name, int i, DirectCompute::Tensor& t, ePostProcessing pp = ePostProcessing::None )
		{
			tempString.Format( "encoder.blocks.%i.%s", i, name );
			map[ tempString ] = PendingTensor{ t, pp };
		};
		auto add2 = [ & ]( const char* name, int i, DirectCompute::TensorPair& t, ePostProcessing ppWeight = ePostProcessing::None, ePostProcessing ppBias = ePostProcessing::None )
		{
			tempString.Format( "encoder.blocks.%i.%s.weight", i, name );
			map[ tempString ] = PendingTensor{ t.w, ppWeight };
			tempString.Format( "encoder.blocks.%i.%s.bias", i, name );
			map[ tempString ] = PendingTensor{ t.b, ppBias };
		};

		for( int i = 0; i < layersEnc; i++ )
		{
			auto& gpu = enc.layers[ i ];
			add2( "mlp_ln", i, gpu.mlpLn );
			add2( "mlp.0", i, gpu.mlp0, ePostProcessing::MakePanels );
			add2( "mlp.2", i, gpu.mlp1, ePostProcessing::MakePanels );
			add2( "attn_ln", i, gpu.attnLn0 );
			add2( "attn.query", i, gpu.attnQuery, ePostProcessing::MakePanels );
			add( "attn.key.weight", i, gpu.attnKey, ePostProcessing::MakePanels );
			add2( "attn.value", i, gpu.attnValue, ePostProcessing::MakePanels );
			add2( "attn.out", i, gpu.attnLn1, ePostProcessing::MakePanels );
		}
	}

	void populateDecodeTensorsMap( CAtlMap<CStringA, PendingTensor>& map, int layersDec, DirectCompute::ModelBuffers& tensors, bool hybrid )
	{
		tensors.dec.layers.resize( layersDec );
		CStringA tempString;
		// Decoder tensors

		auto& dec = tensors.dec;
		if( !hybrid )
		{
			map[ "decoder.positional_embedding" ] = dec.positionalEmbedding;
			map[ "decoder.token_embedding.weight" ] = dec.tokenEmbedding;
			map[ "decoder.ln.weight" ] = dec.ln.w;
			map[ "decoder.ln.bias" ] = dec.ln.b;
		}

		auto add = [ & ]( const char* name, int i, DirectCompute::Tensor& t, ePostProcessing pp = ePostProcessing::None )
		{
			tempString.Format( "decoder.blocks.%i.%s", i, name );
			map[ tempString ] = PendingTensor{ t, pp };
		};
		auto add2 = [ & ]( const char* name, int i, DirectCompute::TensorPair& t, ePostProcessing ppWeight = ePostProcessing::None, ePostProcessing ppBias = ePostProcessing::None )
		{
			tempString.Format( "decoder.blocks.%i.%s.weight", i, name );
			map[ tempString ] = PendingTensor{ t.w, ppWeight };
			tempString.Format( "decoder.blocks.%i.%s.bias", i, name );
			map[ tempString ] = PendingTensor{ t.b, ppBias };
		};

		for( int i = 0; i < layersDec; i++ )
		{
			auto& gpu = dec.layers[ i ];
			add( "cross_attn.key.weight", i, gpu.crossAttnKey, ePostProcessing::MakePanels );
			add2( "cross_attn.value", i, gpu.crossAttnValue, ePostProcessing::MakePanels );
			if( hybrid )
				continue;

			add2( "mlp_ln", i, gpu.mlpLn );
			add2( "mlp.0", i, gpu.mlp0, ePostProcessing::MakePanels );
			add2( "mlp.2", i, gpu.mlp1, ePostProcessing::MakePanels );
			add2( "attn_ln", i, gpu.attnLn0 );
			add2( "attn.query", i, gpu.attnQuery );
			add( "attn.key.weight", i, gpu.attnKey );
			add2( "attn.value", i, gpu.attnValue );
			add2( "attn.out", i, gpu.attnLn1 );
			add2( "cross_attn_ln", i, gpu.crossAttnLn0 );
			add2( "cross_attn.query", i, gpu.crossAttnQuery );
			add2( "cross_attn.out", i, gpu.crossAttnLn1 );
		}
	}

	void populateTensorsMap( CAtlMap<CStringA, PendingTensor>& map, int layersEnc, int layersDec, DirectCompute::ModelBuffers& tensors, bool hybrid )
	{
		populateEncodeTensorsMap( map, layersEnc, tensors );
		populateDecodeTensorsMap( map, layersDec, tensors, hybrid );
	}

	struct sTensorHeader
	{
		int n_dims, length, ftype;
	};

	// compare signed int32 lanes for a <= b
	inline __m128i cmple( __m128i a, __m128i b )
	{
		__m128i i = _mm_min_epi32( a, b );
		return _mm_cmpeq_epi32( a, i );
	}

	inline bool allPositive( const std::array<int, 4>& ne )
	{
		const __m128i v = _mm_loadu_si128( ( const __m128i* )ne.data() );
		const __m128i le = cmple( v, _mm_setzero_si128() );
		return (bool)_mm_testz_si128( le, le );
	}

	inline const char* cstr( const CStringA& s ) { return s; }
}

class WhisperModel::CallbacksImpl : public CpuCompute::iLoaderProgressSink
{
	sLoadModelCallbacks lmcb;
	int64_t fileSize;

	HRESULT gotBytes( int64_t cb ) override final
	{
		if( nullptr != lmcb.cancel )
		{
			HRESULT hr = lmcb.cancel( lmcb.pv );
			CHECK( hr );
			if( S_OK != hr )
				return HRESULT_FROM_WIN32( ERROR_CANCELLED );
		}

		if( nullptr != lmcb.progress )
		{
			postponedBytes -= cb;
			assert( postponedBytes >= 0 );
			int64_t pos = fileSize - postponedBytes;
			const double progressVal = (double)pos / (double)fileSize;
			HRESULT hr = lmcb.progress( progressVal, lmcb.pv );
			CHECK( hr );
		}
		return S_OK;
	}
public:
	int64_t postponedBytes;

	CallbacksImpl()
	{
		lmcb.progress = nullptr;
		lmcb.cancel = nullptr;
		lmcb.pv = nullptr;
		fileSize = 0;
		postponedBytes = 0;
	}

	HRESULT initialize( ComLight::iReadStream* stm, const sLoadModelCallbacks* rsi )
	{
		if( nullptr == rsi )
			return S_OK;
		lmcb = *rsi;
		if( nullptr != lmcb.progress )
			CHECK( stm->getLength( fileSize ) );
		return S_OK;
	}

	HRESULT call( ComLight::iReadStream* stm )
	{
		if( nullptr != lmcb.cancel )
		{
			HRESULT hr = lmcb.cancel( lmcb.pv );
			CHECK( hr );
			if( S_OK != hr )
				return HRESULT_FROM_WIN32( ERROR_CANCELLED );
		}

		if( nullptr != lmcb.progress )
		{
			int64_t pos;
			CHECK( stm->getPosition( pos ) );
			pos -= postponedBytes;
			const double progressVal = (double)pos / (double)fileSize;
			HRESULT hr = lmcb.progress( progressVal, lmcb.pv );
			CHECK( hr );
		}
		return S_OK;
	}
};

HRESULT WhisperModel::loadGpu( ComLight::iReadStream* stm, CallbacksImpl& callbacks )
{
	CAtlMap<CStringA, PendingTensor> map;
	populateTensorsMap( map, parameters.n_audio_layer, parameters.n_text_layer, tensors, false );

	DirectCompute::Reshaper reshape;

	std::vector<uint8_t> bytesVector;
	size_t countLoaded = 0;
	CStringA name;
	int64_t cb = 0;
	while( true )
	{
		CHECK( callbacks.call( stm ) );

		sTensorHeader header;
		HRESULT hr = readStruct( stm, header );
		if( hr == E_EOF )
			break;
		if( FAILED( hr ) )
			return hr;
		if( header.n_dims < 1 || header.n_dims>3 )
			return E_INVALIDARG;

		std::array<int, 4> ne = { 1, 1, 1, 1 };
		CHECK( readBytes( stm, ne.data(), header.n_dims * 4 ) );
		if( !allPositive( ne ) )
			return E_INVALIDARG;

		char* nameBuffer = name.GetBufferSetLength( header.length );
		hr = readBytes( stm, nameBuffer, header.length );
		name.ReleaseBuffer();
		if( FAILED( hr ) )
			return hr;

		auto p = map.Lookup( name );
		if( nullptr == p )
		{
			logError( u8"%s: unknown tensor '%s' in model file", __func__, cstr( name ) );
			return E_INVALIDARG;
		}

		DirectCompute::eDataType dt;
		size_t cbElement;
		if( header.ftype == 0 )
		{
			dt = DirectCompute::eDataType::FP32;
			cbElement = 4;
		}
		else
		{
			dt = DirectCompute::eDataType::FP16;
			cbElement = 2;
		}

		const size_t totalElts = (size_t)(uint32_t)ne[ 0 ] * (uint32_t)ne[ 1 ] * (uint32_t)ne[ 2 ];
		if( totalElts * cbElement > UINT_MAX )
			return DISP_E_OVERFLOW;

		try
		{
			bytesVector.resize( cbElement * totalElts );
		}
		catch( const std::bad_alloc& )
		{
			return E_OUTOFMEMORY;
		}
		CHECK( readBytes( stm, bytesVector.data(), bytesVector.size() ) );
		cb += bytesVector.size();
		CHECK( p->m_value.dest->createImmutable( dt, ne, bytesVector.data() ) );
		CHECK( p->m_value.postProcess( reshape, dt ) );
		countLoaded++;
	}

	if( countLoaded != map.GetCount() )
	{
		logError( u8"Not all tensors loaded from model file - expected %zu, got %zu", map.GetCount(), countLoaded );
		return E_INVALIDARG;
	}

	constexpr double mulMb = 1.0 / ( 1 << 20 );
	logDebug( u8"Loaded %zu GPU tensors, %g MB VRAM", countLoaded, mulMb * cb );
	return S_OK;
}

#if BUILD_HYBRID_VERSION
HRESULT WhisperModel::loadHybrid( ComLight::iReadStream* stm, CallbacksImpl& callbacks )
{
	CAtlMap<CStringA, PendingTensor> map;
	populateTensorsMap( map, parameters.n_audio_layer, parameters.n_text_layer, tensors, true );
	DirectCompute::Reshaper reshape;
	CpuCompute::HybridLoader loader( shared->hybridTensors, parameters.n_text_layer );

	std::vector<uint8_t> bytesVector;
	size_t countLoaded = 0;
	CStringA name;
	int64_t cb = 0;
	while( true )
	{
		CHECK( callbacks.call( stm ) );

		sTensorHeader header;
		HRESULT hr = readStruct( stm, header );
		if( hr == E_EOF )
			break;
		if( FAILED( hr ) )
			return hr;
		if( header.n_dims < 1 || header.n_dims > 3 )
			return E_INVALIDARG;

		std::array<int, 4> ne = { 1, 1, 1, 1 };
		CHECK( readBytes( stm, ne.data(), header.n_dims * 4 ) );
		if( !allPositive( ne ) )
			return E_INVALIDARG;

		char* nameBuffer = name.GetBufferSetLength( header.length );
		hr = readBytes( stm, nameBuffer, header.length );
		name.ReleaseBuffer();
		if( FAILED( hr ) )
			return hr;

		auto p = map.Lookup( name );
		if( nullptr == p )
		{
			HRESULT hr = loader.setupTensor( name, header.n_dims, header.ftype, ne, stm, callbacks.postponedBytes );
			if( hr == S_OK )
				continue;
			logError( u8"%s: unknown tensor '%s' in model file", __func__, cstr( name ) );
			return E_INVALIDARG;
		}

		DirectCompute::eDataType dt;
		size_t cbElement;
		if( header.ftype == 0 )
		{
			dt = DirectCompute::eDataType::FP32;
			cbElement = 4;
		}
		else
		{
			dt = DirectCompute::eDataType::FP16;
			cbElement = 2;
		}

		const size_t totalElts = (size_t)(uint32_t)ne[ 0 ] * (uint32_t)ne[ 1 ] * (uint32_t)ne[ 2 ];
		if( totalElts * cbElement > UINT_MAX )
			return DISP_E_OVERFLOW;

		try
		{
			bytesVector.resize( cbElement * totalElts );
		}
		catch( const std::bad_alloc& )
		{
			return E_OUTOFMEMORY;
		}
		CHECK( readBytes( stm, bytesVector.data(), bytesVector.size() ) );
		CHECK( p->m_value.dest->createImmutable( dt, ne, bytesVector.data() ) );
		CHECK( p->m_value.postProcess( reshape, dt ) );
		countLoaded++;
		cb += bytesVector.size();
	}

	if( countLoaded != map.GetCount() )
	{
		logError( u8"Not all tensors loaded from model file - expected %zu, got %zu", map.GetCount(), countLoaded );
		return E_INVALIDARG;
	}

	constexpr double mulMb = 1.0 / ( 1 << 20 );
	logDebug( u8"Loaded %zu GPU tensors, %g MB VRAM", countLoaded, mulMb * cb );

	CHECK( loader.completeLoad( stm, callbacks ) );
	return S_OK;
}
#endif

HRESULT WhisperModel::load( ComLight::iReadStream* stm, bool hybrid, const sLoadModelCallbacks* callbacks )
{
	CpuProfiler cpuPerf;
	CallbacksImpl cb;
	CHECK( cb.initialize( stm, callbacks ) );
	// verify magic
	{
		uint32_t magic;
		CHECK( readStruct( stm, magic ) );
		if( magic != 0x67676d6c )
		{
			logError( u8"Invalid model file, bad magic" );
			return E_INVALIDARG;
		}
	}

	shared = std::make_shared<ModelShared>();

	// hparams and MEL filters
	{
		ParamsAndMelHeader pmh;
		CHECK( readStruct( stm, pmh ) );
		parameters = pmh.mp;
		assert( parameters.n_text_state == parameters.n_audio_state );

		shared->filters.n_mel = pmh.n_mel;
		shared->filters.n_fft = pmh.n_fft;
		const size_t len = (size_t)pmh.n_mel * pmh.n_fft;
		shared->filters.data.resize( len );
		CHECK( readBytes( stm, shared->filters.data.data(), len * 4 ) );

		const int64_t cb = len * 4;
		constexpr double mulKb = 1.0 / ( 1 << 10 );
		logDebug( u8"Loaded MEL filters, %.1f kb RAM", mulKb * cb );
	}
	CHECK( cb.call( stm ) );

	// Vocabulary
	CHECK( shared->vocab.load( stm, parameters.n_vocab ) );
	CHECK( cb.call( stm ) );

	DirectCompute::GpuProfilerSimple gpuProfiler;
	CHECK( gpuProfiler.create() );

	if( hybrid )
	{
#if BUILD_HYBRID_VERSION
		CHECK( loadHybrid( stm, cb ) )
#else
		return E_NOTIMPL;
#endif
	}
	else
		CHECK( loadGpu( stm, cb ) );

	CHECK( gpuProfiler.time( loadTimeGpu ) );
	loadTimeCpu = cpuPerf.elapsed();
	return S_OK;
}

HRESULT Whisper::WhisperModel::createClone( const WhisperModel& rsi )
{
	parameters = rsi.parameters;
	shared = rsi.shared;
	CHECK( tensors.createClone( rsi.tensors ) );
	return S_OK;
}

__m128i Whisper::WhisperModel::getMemoryUse() const
{
	size_t cb = shared->vocab.getMemoryUse();
	cb += vectorMemoryUse( shared->filters.data );
	__m128i v = _mm_cvtsi64_si128( (int64_t)cb );
	v = _mm_add_epi64( v, tensors.getMemoryUse() );
	return v;
}