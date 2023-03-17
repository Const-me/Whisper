#include "stdafx.h"
#include "createDevice.h"
#include "listGPUs.h"
#include "RenderDoc/renderDoc.h"
#pragma comment(lib, "D3D11.lib")
#include <atlstr.h>

HRESULT DirectCompute::createDevice( const std::wstring& adapterName, ID3D11Device** dev, ID3D11DeviceContext** context )
{
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
	HRESULT hr = D3D11CreateDevice( adapter, driverType, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, dev, nullptr, context );
	if( SUCCEEDED( hr ) )
		return S_OK;
	// D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT: This value is not supported until Direct3D 11.1
	// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_create_device_flag
	flags = _andn_u32( D3D11_CREATE_DEVICE_DISABLE_GPU_TIMEOUT, flags );

	hr = D3D11CreateDevice( adapter, driverType, nullptr, flags, levels.data(), levelsCount, D3D11_SDK_VERSION, dev, nullptr, context );
	if( SUCCEEDED( hr ) )
		return S_OK;
	return hr;
}

HRESULT DirectCompute::cloneDevice( ID3D11Device* source, ID3D11Device** dev, ID3D11DeviceContext** context )
{
	CComPtr<IDXGIDevice> dxgiDev;
	CHECK( source->QueryInterface( &dxgiDev ) );

	CComPtr<IDXGIAdapter> adapter;
	CHECK( dxgiDev->GetAdapter( &adapter ) );

	const uint32_t flags = source->GetCreationFlags();
	const D3D_FEATURE_LEVEL level = source->GetFeatureLevel();
	return D3D11CreateDevice( adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, &level, 1,
		D3D11_SDK_VERSION, dev, nullptr, context );
}

namespace
{
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
}

HRESULT DirectCompute::validateFlags( uint32_t flags )
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

HRESULT DirectCompute::queryDeviceInfo( sGpuInfo& rdi, ID3D11Device* dev, uint32_t flags )
{
	if( nullptr == dev )
		return OLE_E_BLANK;

	CComPtr<IDXGIDevice> dd;
	CHECK( dev->QueryInterface( &dd ) );

	CComPtr<IDXGIAdapter> adapter;
	CHECK( dd->GetAdapter( &adapter ) );

	DXGI_ADAPTER_DESC desc;
	adapter->GetDesc( &desc );

	const size_t descLen = wcsnlen_s( desc.Description, 128 );
	const wchar_t* rsi = &desc.Description[ 0 ];
	rdi.description.assign( rsi, rsi + descLen );
	rdi.vendor = (eGpuVendor)desc.VendorId;
	rdi.device = (uint16_t)desc.DeviceId;
	rdi.revision = (uint16_t)desc.Revision;
	rdi.subsystem = desc.SubSysId;
	rdi.vramDedicated = desc.DedicatedVideoMemory;
	rdi.ramDedicated = desc.DedicatedSystemMemory;
	rdi.ramShared = desc.SharedSystemMemory;

	// Set up these flags
	uint8_t ef = 0;
	const bool amd = ( rdi.vendor == eGpuVendor::AMD );
	if( merge3( flags, eGpuModelFlags::Wave64, eGpuModelFlags::Wave32, amd ) )
		ef |= (uint8_t)eGpuEffectiveFlags::Wave64;
	if( merge3( flags, eGpuModelFlags::UseReshapedMatMul, eGpuModelFlags::NoReshapedMatMul, amd ) )
		ef |= (uint8_t)eGpuEffectiveFlags::ReshapedMatMul;
	if( 0 != ( flags & eGpuModelFlags::Cloneable ) )
		ef |= (uint8_t)eGpuEffectiveFlags::Cloneable;
	rdi.flags = (eGpuEffectiveFlags)ef;

	if( willLogMessage( Whisper::eLogLevel::Debug ) )
	{
		const int fl = dev->GetFeatureLevel();
		const int flMajor = ( fl >> 12 ) & 0xF;
		const int flMinor = ( fl >> 8 ) & 0xF;

		CStringA flagsString;
		flagsString.Format( "%s | %s", rdi.wave64() ? "Wave64" : "Wave32",
			rdi.useReshapedMatMul() ? "UseReshapedMatMul" : "NoReshapedMatMul" );
		if( rdi.cloneableModel() )
			flagsString += " | Cloneable";

		logDebug16( L"Using GPU \"%s\", feature level %i.%i, effective flags %S",
			rdi.description.c_str(), flMajor, flMinor,
			flagsString.operator const char* ( ) );
	}
	return S_OK;
}