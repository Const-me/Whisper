#include "stdafx.h"
#include "HybridLoader.h"
using namespace CpuCompute;
using namespace ComLight;

static void populateDecodeTensorsMap( CAtlMap<CStringA, Tensor*>& map, int layersDec, DecoderTensors& dec )
{
	dec.layers.resize( layersDec );

	map[ "decoder.positional_embedding" ] = &dec.positionalEmbedding;
	map[ "decoder.token_embedding.weight" ] = &dec.tokenEmbedding;
	map[ "decoder.ln.weight" ] = &dec.ln.w;
	map[ "decoder.ln.bias" ] = &dec.ln.b;

	CStringA tempString;
	auto add = [ & ]( const char* name, int i, Tensor& t )
	{
		tempString.Format( "decoder.blocks.%i.%s", i, name );
		map[ tempString ] = &t;
	};

	auto add2 = [ & ]( const char* name, int i, TensorPair& tensors )
	{
		tempString.Format( "decoder.blocks.%i.%s.weight", i, name );
		map[ tempString ] = &tensors.w;
		tempString.Format( "decoder.blocks.%i.%s.bias", i, name );
		map[ tempString ] = &tensors.b;
	};

	for( int i = 0; i < layersDec; i++ )
	{
		auto& gpu = dec.layers[ i ];
		add2( "mlp_ln", i, gpu.mlpLn );
		add2( "mlp.0", i, gpu.mlp0 );
		add2( "mlp.2", i, gpu.mlp1 );
		add2( "attn_ln", i, gpu.attnLn0 );
		add2( "attn.query", i, gpu.attnQuery );
		add( "attn.key.weight", i, gpu.attnKey );

		add2( "attn.value", i, gpu.attnValue );
		add2( "attn.out", i, gpu.attnLn1 );

		add2( "cross_attn_ln", i, gpu.crossAttnLn0 );
		add2( "cross_attn.query", i, gpu.crossAttnQuery );

		// These 3 tensors are used by the encode() method, to compute cross-attention buffers
		// Need them in VRAM even for the hybrid model
		// add( "cross_attn.key.weight", i, gpu.cross_attn_k_w );
		// add2( "cross_attn.value", i, gpu.cross_attn_v_w, gpu.cross_attn_v_b );
		add2( "cross_attn.out", i, gpu.crossAttnLn1 );
	}
}

HybridLoader::HybridLoader( DecoderTensors& m, int countLayers ) :
	destination( m )
{
	populateDecodeTensorsMap( map, countLayers, destination );
	pending.reserve( map.GetCount() );
}

HRESULT HybridLoader::setupTensor( const CStringA& name, int n_dims, int ftype, const std::array<int, 4>& ne, ComLight::iReadStream* stream, int64_t& postponedBytes )
{
	auto p = map.Lookup( name );
	if( nullptr == p )
		return S_FALSE;

	Tensor& rdi = *p->m_value;
	PendingTensor& pt = pending.emplace_back();

	__m128i vec = load16( ne.data() );
	vec = _mm_insert_epi32( vec, 1, 3 );
	store16( &rdi.ne, vec );
	rdi.setDenseStrides();

	pt.destPointer = p->m_value;
	CHECK( stream->getPosition( pt.streamOffset ) );
	pt.bufferOffset = bufferBytes;

	size_t cbElement;
	if( ftype == 0 )
	{
		rdi.setType( eDataType::FP32 );
		cbElement = 4;
	}
	else
	{
		rdi.setType( eDataType::FP16 );
		cbElement = 2;
	}

	const size_t totalElts = (size_t)(uint32_t)ne[ 0 ] * (uint32_t)ne[ 1 ] * (uint32_t)ne[ 2 ];
	if( totalElts * cbElement > UINT_MAX )
		return DISP_E_OVERFLOW;

	size_t payloadBytes = cbElement * totalElts;
	pt.payloadBytes = payloadBytes;
	CHECK( stream->seek( payloadBytes, eSeekOrigin::Current ) );
	postponedBytes += (int64_t)payloadBytes;

	payloadBytes = ( payloadBytes + 31 ) & ( ~( (size_t)31 ) );
	bufferBytes += payloadBytes;
	return S_OK;
}

HRESULT HybridLoader::completeLoad( ComLight::iReadStream* stream, iLoaderProgressSink& progressSink )
{
	if( pending.size() != map.GetCount() )
	{
		logError( u8"Not all tensors loaded from model file - expected %zu, got %zu", map.GetCount(), pending.size() );
		return E_INVALIDARG;
	}

	LargeBuffer buffer;
	CHECK( buffer.allocate( bufferBytes ) );

	uint8_t* rdi = buffer.pointer();

	for( const auto& pt : pending )
	{
		if( pt.payloadBytes > INT_MAX )
			return DISP_E_OVERFLOW;
		CHECK( stream->seek( pt.streamOffset, eSeekOrigin::Begin ) );

		int written = 0;
		CHECK( stream->read( rdi, (int)pt.payloadBytes, written ) );
		CHECK( progressSink.gotBytes( (int64_t)pt.payloadBytes ) );

		pt.destPointer->setDataPointer( rdi );

		const size_t cb = ( pt.payloadBytes + 31 ) & ( ~( (size_t)31 ) );
		rdi += cb;
	}

	CHECK( buffer.setReadOnly( bufferBytes ) );
	destination.setMemoryBuffer( std::move( buffer ) );

	constexpr double mulMb = 1.0 / ( 1 << 20 );
	logDebug( u8"Loaded %zu decoder tensors, %g MB RAM", pending.size(), mulMb * (double)(int64_t)bufferBytes );
	return S_OK;
}