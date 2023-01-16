#include "stdafx.h"
#include "ModelLoader.h"


#if BUILD_BOTH_VERSIONS
namespace DirectCompute
{
	static ModelBuffers s_model;
	const ModelBuffers& gpuModel = s_model;
}

using namespace DirectCompute;

ModelLoader::ModelLoader( int encoderLayers, int decoderLayers ) :
	model( s_model )
{
	if( encoderLayers <= 0 || decoderLayers <= 0 )
		throw E_INVALIDARG;
	model.enc.layers.resize( (uint32_t)encoderLayers );
	model.dec.layers.resize( (uint32_t)decoderLayers );
}

void ModelLoader::add( const ggml_tensor* ggml, Tensor& gpu )
{
	if( nullptr == ggml )
		throw E_POINTER;

	auto res = map.try_emplace( ggml, &gpu );
	if( !res.second )
		throw E_INVALIDARG;
}

Tensor* ModelLoader::lookup( const ggml_tensor* ggml ) const
{
	auto it = map.find( ggml );
	if( it == map.end() )
		return nullptr;
	return it->second;
}

bool ModelLoader::tryLoad( const ggml_tensor* ggml )
{
	Tensor* rdi = lookup( ggml );
	if( nullptr == rdi )
		return false;
	HRESULT hr = rdi->create( *ggml, eBufferUse::Immutable, true );
	if( SUCCEEDED( hr ) )
		return true;
	throw hr;
}
#endif

__m128i __declspec( noinline ) DirectCompute::TensorPair::getMemoryUse() const
{
	return _mm_add_epi64( w.getMemoryUse(), b.getMemoryUse() );
}

__m128i DirectCompute::LayerEncoder::getMemoryUse() const
{
	__m128i v = attnLn0.getMemoryUse();
	v = _mm_add_epi64( v, attnLn1.getMemoryUse() );
	v = _mm_add_epi64( v, attnQuery.getMemoryUse() );
	v = _mm_add_epi64( v, attnKey.getMemoryUse() );
	v = _mm_add_epi64( v, attnValue.getMemoryUse() );
	v = _mm_add_epi64( v, mlpLn.getMemoryUse() );
	v = _mm_add_epi64( v, mlp0.getMemoryUse() );
	v = _mm_add_epi64( v, mlp1.getMemoryUse() );
	return v;
}

__m128i DirectCompute::EncoderBuffers::getMemoryUse() const
{
	__m128i v = _mm_cvtsi64_si128( vectorMemoryUse( layers ) );
	v = _mm_add_epi64( v, positionalEmbedding.getMemoryUse() );
	v = _mm_add_epi64( v, conv1.getMemoryUse() );
	v = _mm_add_epi64( v, conv2.getMemoryUse() );
	v = _mm_add_epi64( v, lnPost.getMemoryUse() );
	for( const auto& layer : layers )
		v = _mm_add_epi64( v, layer.getMemoryUse() );
	return v;
}

__m128i DirectCompute::LayerDecoder::getMemoryUse() const
{
	__m128i v = attnLn0.getMemoryUse();
	v = _mm_add_epi64( v, attnLn1.getMemoryUse() );
	v = _mm_add_epi64( v, attnQuery.getMemoryUse() );
	v = _mm_add_epi64( v, attnKey.getMemoryUse() );
	v = _mm_add_epi64( v, attnValue.getMemoryUse() );
	v = _mm_add_epi64( v, crossAttnLn0.getMemoryUse() );
	v = _mm_add_epi64( v, crossAttnLn1.getMemoryUse() );
	v = _mm_add_epi64( v, crossAttnQuery.getMemoryUse() );
	v = _mm_add_epi64( v, crossAttnKey.getMemoryUse() );
	v = _mm_add_epi64( v, crossAttnValue.getMemoryUse() );
	v = _mm_add_epi64( v, mlpLn.getMemoryUse() );
	v = _mm_add_epi64( v, mlp0.getMemoryUse() );
	v = _mm_add_epi64( v, mlp1.getMemoryUse() );
	return v;
}

__m128i DirectCompute::DecoderBuffers::getMemoryUse() const
{
	__m128i v = _mm_cvtsi64_si128( vectorMemoryUse( layers ) );
	v = _mm_add_epi64( v, positionalEmbedding.getMemoryUse() );
	v = _mm_add_epi64( v, tokenEmbedding.getMemoryUse() );
	v = _mm_add_epi64( v, ln.getMemoryUse() );
	for( const auto& layer : layers )
		v = _mm_add_epi64( v, layer.getMemoryUse() );
	return v;
}

__m128i DirectCompute::ModelBuffers::getMemoryUse() const
{
	return _mm_add_epi64( enc.getMemoryUse(), dec.getMemoryUse() );
}