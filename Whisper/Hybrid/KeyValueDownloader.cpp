#include "stdafx.h"
#include "KeyValueDownloader.h"

HRESULT KeyValueDownloader::create( const Whisper::sModelParams& mp )
{
	const uint32_t n_audio_ctx = mp.n_audio_ctx;
	const uint32_t n_mem = mp.n_text_layer * mp.n_audio_ctx;
	const uint32_t n_elements = mp.n_text_state * n_mem;

	CD3D11_BUFFER_DESC desc{ n_elements * 2, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ };
	ID3D11Device* dev = DirectCompute::device();
	CHECK( dev->CreateBuffer( &desc, nullptr, &keys ) );
	CHECK( dev->CreateBuffer( &desc, nullptr, &values ) );

	length = n_elements;
	return S_OK;
}

HRESULT KeyValueDownloader::download( const DirectCompute::KeyValueBuffers& source )
{
	ID3D11DeviceContext* ctx = DirectCompute::context();
	ctx->CopyResource( keys, source.keys.getBuffer() );
	ctx->CopyResource( values, source.values.getBuffer() );
	return S_OK;
}

KeyValueDownloader::ReadMap::ReadMap( KeyValueDownloader& owner ) :
	length( owner.length )
{
	check( mappedKeys.map( owner.keys, true ) );
	check( mappedValues.map( owner.values, true ) );
}