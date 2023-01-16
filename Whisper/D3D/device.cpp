#include "stdafx.h"
#include "device.h"
#include <immintrin.h>
#include <ammintrin.h>
#pragma comment(lib, "D3D11.lib")
#include "RenderDoc/renderDoc.h"

namespace DirectCompute
{
	CComPtr<ID3D11Device> g_device;
	CComPtr<ID3D11DeviceContext> g_context;
	D3D_FEATURE_LEVEL g_featureLevel = (D3D_FEATURE_LEVEL)0;

	ID3D11Device* device() { return g_device; }
	ID3D11DeviceContext* context() { return g_context; }
	D3D_FEATURE_LEVEL featureLevel() { return g_featureLevel; }

	void terminate()
	{
		g_context = nullptr;
		g_device = nullptr;
	}

	static HRESULT createDevice()
	{
		if( g_device )
			return S_FALSE;

		const std::array<D3D_FEATURE_LEVEL, 4> levels = { D3D_FEATURE_LEVEL_12_1 , D3D_FEATURE_LEVEL_12_0 , D3D_FEATURE_LEVEL_11_1 , D3D_FEATURE_LEVEL_11_0 };
		UINT flags = D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT | D3D11_CREATE_DEVICE_SINGLETHREADED;
		bool renderDoc = initializeRenderDoc();
#ifdef _DEBUG
		if( !renderDoc )
		{
			// Last time I checked, RenderDoc crashed with debug version of D3D11 runtime
			// Only setting this flag unless renderdoc.dll is loaded to the current process
			flags |= D3D11_CREATE_DEVICE_DEBUG;
		}
#endif
		constexpr UINT levelsCount = (UINT)levels.size();
		HRESULT hr = D3D11CreateDevice( nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, &g_device, &g_featureLevel, &g_context );
		if( SUCCEEDED( hr ) )
			return S_OK;
		// D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT: This value is not supported until Direct3D 11.1
		// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_create_device_flag
		flags = _andn_u32( D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT, flags );

		hr = D3D11CreateDevice( nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, &g_device, &g_featureLevel, &g_context );
		if( SUCCEEDED( hr ) )
			return S_OK;
		return hr;
	}

	sGpuInfo s_gpuInfo = {};
	const sGpuInfo& gpuInfo = s_gpuInfo;

	static HRESULT queryDeviceInfo()
	{
		if( nullptr == g_device )
			return OLE_E_BLANK;
		CComPtr<IDXGIDevice> dd;
		CHECK( g_device.QueryInterface( &dd ) );

		CComPtr<IDXGIAdapter> adapter;
		CHECK( dd->GetAdapter( &adapter ) );

		DXGI_ADAPTER_DESC desc;
		adapter->GetDesc( &desc );

		const size_t descLen = wcsnlen_s( desc.Description, 128 );
		const wchar_t* rsi = &desc.Description[ 0 ];
		s_gpuInfo.description.assign( rsi, rsi + descLen );
		s_gpuInfo.vendor = (eGpuVendor)desc.VendorId;
		s_gpuInfo.device = (uint16_t)desc.DeviceId;
		s_gpuInfo.revision = (uint16_t)desc.Revision;
		s_gpuInfo.subsystem = desc.SubSysId;
		s_gpuInfo.vramDedicated = desc.DedicatedVideoMemory;
		s_gpuInfo.ramDedicated = desc.DedicatedSystemMemory;
		s_gpuInfo.ramShared = desc.SharedSystemMemory;
		return S_OK;
	}

	HRESULT initialize()
	{
		HRESULT hr = createDevice();
		if( hr != S_OK )
			return hr;
		queryDeviceInfo();
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