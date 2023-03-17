#include "stdafx.h"
#include "device.h"
#include "../API/sModelSetup.h"
#include "createDevice.h"

namespace DirectCompute
{
	CComPtr<ID3D11Device> g_device;
	CComPtr<ID3D11DeviceContext> g_context;
	D3D_FEATURE_LEVEL g_featureLevel = (D3D_FEATURE_LEVEL)0;

	ID3D11Device* device() { return g_device; }
	ID3D11DeviceContext* context() { return g_context; }

	void terminate()
	{
		g_context = nullptr;
		g_device = nullptr;
	}

	inline HRESULT createDevice( const std::wstring& adapterName )
	{
		if( g_device )
			return S_FALSE;
		return createDevice( adapterName, &g_device, &g_context );
	}

	sGpuInfo s_gpuInfo = {};
	const sGpuInfo& gpuInfo() { return s_gpuInfo; }

	HRESULT initialize( uint32_t flags, const std::wstring& adapter )
	{
		CHECK( validateFlags( flags ) );
		HRESULT hr = createDevice( adapter );
		if( hr != S_OK )
			return hr;
		queryDeviceInfo( s_gpuInfo, g_device, flags );
		return S_OK;
	}

	__m128i __declspec( noinline ) bufferMemoryUsage( ID3D11Buffer* buffer )
	{
		if( nullptr != buffer )
		{
			D3D11_BUFFER_DESC desc;
			buffer->GetDesc( &desc );

			if( desc.Usage != D3D11_USAGE_STAGING )
				return setHigh_size( desc.ByteWidth );
			else
				return setLow_size( desc.ByteWidth );
		}
		return _mm_setzero_si128();
	}

	__m128i __declspec( noinline ) resourceMemoryUsage( ID3D11ShaderResourceView* srv )
	{
		if( nullptr != srv )
		{
			CComPtr<ID3D11Resource> res;
			srv->GetResource( &res );
			CComPtr<ID3D11Buffer> buff;
			if( SUCCEEDED( res.QueryInterface( &buff ) ) )
				return bufferMemoryUsage( buff );
			assert( false );	// We don't use textures in this project
		}
		return _mm_setzero_si128();
	}
}