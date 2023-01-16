#include "stdafx.h"
#include "downloadBuffer.h"
#include "device.h"
#include "MappedResource.h"

namespace
{
	struct BufferInfo
	{
		D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
		D3D11_BUFFER_DESC bufferDesc;
		CComPtr<ID3D11Buffer> source;

		HRESULT create( ID3D11ShaderResourceView* srv )
		{
			srv->GetDesc( &viewDesc );
			if( viewDesc.ViewDimension != D3D_SRV_DIMENSION_BUFFER )
				return E_INVALIDARG;

			CComPtr<ID3D11Resource> res;
			srv->GetResource( &res );
			CHECK( res.QueryInterface( &source ) );

			source->GetDesc( &bufferDesc );
			return S_OK;
		}

		HRESULT download( void* rdi )
		{
			bufferDesc.BindFlags = 0;
			bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			bufferDesc.Usage = D3D11_USAGE_STAGING;
			CComPtr<ID3D11Buffer> staging;
			using namespace DirectCompute;
			CHECK( device()->CreateBuffer( &bufferDesc, nullptr, &staging ) );

			context()->CopyResource( staging, source );

			MappedResource mapped;
			mapped.map( staging, true );
			memcpy( rdi, mapped.data(), bufferDesc.ByteWidth );
			return S_OK;
		}
	};

	size_t dxgiSizeof( DXGI_FORMAT fmt )
	{
		switch( fmt )
		{
		case DXGI_FORMAT_R16_FLOAT: return 2;
		case DXGI_FORMAT_R32_FLOAT: return 4;
		}
		return 0;
	}
}

template<class E>
HRESULT DirectCompute::downloadBuffer( ID3D11ShaderResourceView* srv, std::vector<E>& vec )
{
	BufferInfo bi;
	CHECK( bi.create( srv ) );

	const size_t cb = dxgiSizeof( bi.viewDesc.Format );
	if( cb != sizeof( E ) )
		return E_INVALIDARG;

	vec.resize( bi.bufferDesc.ByteWidth / cb );
	return bi.download( vec.data() );
}

template HRESULT DirectCompute::downloadBuffer( ID3D11ShaderResourceView* srv, std::vector<uint16_t>& vec );
template HRESULT DirectCompute::downloadBuffer( ID3D11ShaderResourceView* srv, std::vector<float>& vec );