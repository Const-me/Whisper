#include "stdafx.h"
#include "device.h"
#include <immintrin.h>
#include <ammintrin.h>
#pragma comment(lib, "D3D11.lib")
#include "RenderDoc/renderDoc.h"
#include "../API/sModelSetup.h"
#include "listGPUs.h"

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

	static HRESULT createDevice( const std::wstring& adapterName )
	{
		if( g_device )
			return S_FALSE;

		CComPtr<IDXGIAdapter1> adapter = selectAdapter( adapterName );
		const D3D_DRIVER_TYPE driverType = adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;

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
		HRESULT hr = D3D11CreateDevice( adapter, driverType, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, &g_device, &g_featureLevel, &g_context );
		if( SUCCEEDED( hr ) )
			return S_OK;
		// D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT: This value is not supported until Direct3D 11.1
		// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_create_device_flag
		flags = _andn_u32( D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT, flags );

		hr = D3D11CreateDevice( adapter, driverType, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, &g_device, &g_featureLevel, &g_context );
		if( SUCCEEDED( hr ) )
			return S_OK;
		return hr;
	}

	sGpuInfo s_gpuInfo = {};
	const sGpuInfo& gpuInfo = s_gpuInfo;

	using Whisper::eGpuModelFlags;
	inline constexpr uint32_t operator|( eGpuModelFlags a, eGpuModelFlags b )
	{
		return (uint32_t)a | (uint32_t)b;
	}
	inline bool operator&( uint32_t flags, eGpuModelFlags bit )
	{
		return 0 != ( flags & (uint32_t)bit );
	}
	inline bool merge3( uint32_t flags, eGpuModelFlags enabled, eGpuModelFlags disabled, bool def )
	{
		if( flags & enabled )
			return true;
		if( flags & disabled )
			return false;
		return def;
	}

	static HRESULT queryDeviceInfo( uint32_t flags )
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

		// Set up these flags
		uint8_t ef = 0;
		const bool amd = ( s_gpuInfo.vendor == eGpuVendor::AMD );
		if( merge3( flags, eGpuModelFlags::Wave64, eGpuModelFlags::Wave32, amd ) )
			ef |= (uint8_t)eGpuEffectiveFlags::Wave64;
		if( merge3( flags, eGpuModelFlags::UseReshapedMatMul, eGpuModelFlags::NoReshapedMatMul, amd ) )
			ef |= (uint8_t)eGpuEffectiveFlags::ReshapedMatMul;
		s_gpuInfo.flags = (eGpuEffectiveFlags)ef;


		if( willLogMessage( Whisper::eLogLevel::Debug ) )
		{
			const int fl = g_featureLevel;
			const int flMajor = ( fl >> 12 ) & 0xF;
			const int flMinor = ( fl >> 8 ) & 0xF;

			logDebug16( L"Using GPU \"%s\", feature level %i.%i, effective flags %S | %S",
				s_gpuInfo.description.c_str(), flMajor, flMinor,
				s_gpuInfo.wave64() ? "Wave64" : "Wave32",
				s_gpuInfo.useReshapedMatMul() ? "UseReshapedMatMul" : "NoReshapedMatMul" );
		}
		return S_OK;
	}

	static HRESULT validateFlags( uint32_t flags )
	{
		constexpr uint32_t waveBoth = eGpuModelFlags::Wave32 | eGpuModelFlags::Wave64;
		if( ( flags & waveBoth ) == waveBoth )
		{
			logError( u8"eGpuModelFlags.%s and eGpuModelFlags.%s are mutually exclusive", "Wave32", "Wave64" );
			return E_INVALIDARG;
		}

		constexpr uint32_t reshapedBoth = eGpuModelFlags::NoReshapedMatMul | eGpuModelFlags::UseReshapedMatMul;
		if( ( flags & reshapedBoth ) == reshapedBoth )
		{
			logError( u8"eGpuModelFlags.%s and eGpuModelFlags.%s are mutually exclusive", "NoReshapedMatMul", "UseReshapedMatMul" );
			return E_INVALIDARG;
		}
		return S_OK;
	}

	HRESULT initialize( uint32_t flags, const std::wstring& adapter )
	{
		CHECK( validateFlags( flags ) );
		HRESULT hr = createDevice( adapter );
		if( hr != S_OK )
			return hr;
		queryDeviceInfo( flags );
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