#include <stdafx.h>
#include "ModelBuffers.h"
#include "../ML/mlUtils.h"

namespace
{
	using namespace DirectCompute;

	HRESULT clone( Tensor& rdi, const Tensor& rsi )
	{
		CComPtr<ID3D11ShaderResourceView> srv;
		CHECK( cloneResourceView( rsi, &srv ) );

		rdi = rsi;
		rdi.setGpuViews( srv );
		return S_OK;
	}

	HRESULT clone( TensorPair& rdi, const TensorPair& rsi )
	{
		CHECK( clone( rdi.w, rsi.w ) );
		CHECK( clone( rdi.b, rsi.b ) );
		return S_OK;
	}

	HRESULT clone( LayerEncoder& rdi, const LayerEncoder& rsi )
	{
		CHECK( clone( rdi.attnLn0, rsi.attnLn0 ) );
		CHECK( clone( rdi.attnLn1, rsi.attnLn1 ) );
		CHECK( clone( rdi.attnQuery, rsi.attnQuery ) );
		CHECK( clone( rdi.attnKey, rsi.attnKey ) );
		CHECK( clone( rdi.attnValue, rsi.attnValue ) );
		CHECK( clone( rdi.mlpLn, rsi.mlpLn ) );
		CHECK( clone( rdi.mlp0, rsi.mlp0 ) );
		CHECK( clone( rdi.mlp1, rsi.mlp1 ) );

		return S_OK;
	}
	HRESULT clone( LayerDecoder& rdi, const LayerDecoder& rsi )
	{
		CHECK( clone( rdi.attnLn0, rsi.attnLn0 ) );
		CHECK( clone( rdi.attnLn1, rsi.attnLn1 ) );
		CHECK( clone( rdi.attnQuery, rsi.attnQuery ) );
		CHECK( clone( rdi.attnKey, rsi.attnKey ) );
		CHECK( clone( rdi.attnValue, rsi.attnValue ) );
		CHECK( clone( rdi.crossAttnLn0, rsi.crossAttnLn0 ) );
		CHECK( clone( rdi.crossAttnLn1, rsi.crossAttnLn1 ) );
		CHECK( clone( rdi.crossAttnQuery, rsi.crossAttnQuery ) );
		CHECK( clone( rdi.crossAttnKey, rsi.crossAttnKey ) );
		CHECK( clone( rdi.crossAttnValue, rsi.crossAttnValue ) );
		CHECK( clone( rdi.mlpLn, rsi.mlpLn ) );
		CHECK( clone( rdi.mlp0, rsi.mlp0 ) );
		CHECK( clone( rdi.mlp1, rsi.mlp1 ) );

		return S_OK;
	}

	template<class E>
	HRESULT clone( std::vector<E>& rdi, const std::vector<E>& rsi )
	{
		const size_t len = rsi.size();
		try
		{
			rdi.resize( len );
		}
		catch( const std::bad_alloc& )
		{
			return E_OUTOFMEMORY;
		}

		for( size_t i = 0; i < len; i++ )
			CHECK( clone( rdi[ i ], rsi[ i ] ) );

		return S_OK;
	}

	HRESULT clone( EncoderBuffers& rdi, const EncoderBuffers& rsi )
	{
		CHECK( clone( rdi.layers, rsi.layers ) );
		CHECK( clone( rdi.positionalEmbedding, rsi.positionalEmbedding ) );
		CHECK( clone( rdi.conv1, rsi.conv1 ) );
		CHECK( clone( rdi.conv2, rsi.conv2 ) );
		CHECK( clone( rdi.lnPost, rsi.lnPost ) );
		return S_OK;
	}

	HRESULT clone( DecoderBuffers& rdi, const DecoderBuffers& rsi )
	{
		CHECK( clone( rdi.positionalEmbedding, rsi.positionalEmbedding ) );
		CHECK( clone( rdi.tokenEmbedding, rsi.tokenEmbedding ) );
		CHECK( clone( rdi.ln, rsi.ln ) );
		CHECK( clone( rdi.layers, rsi.layers ) );

		return S_OK;
	}
}

HRESULT ModelBuffers::createClone( const ModelBuffers& rsi )
{
	CHECK( clone( enc, rsi.enc ) );
	CHECK( clone( dec, rsi.dec ) );

	return S_OK;
}