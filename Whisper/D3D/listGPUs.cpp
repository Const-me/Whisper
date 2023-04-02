#include "stdafx.h"
#include "listGPUs.h"
#pragma comment(lib, "DXGI.lib")
#include <charconv>
#include <optional>

namespace DirectCompute
{
	static HRESULT createFactory( CComPtr<IDXGIFactory1>& rdi )
	{
		HRESULT hr = CreateDXGIFactory1( IID_PPV_ARGS( &rdi ) );
		if( SUCCEEDED( hr ) )
			return S_OK;
		return hr;
	}

	inline void setName( std::wstring& rdi, const DXGI_ADAPTER_DESC1& desc )
	{
		const size_t descLen = wcsnlen_s( desc.Description, 128 );
		const wchar_t* rsi = &desc.Description[ 0 ];
		rdi.assign( rsi, rsi + descLen );
	}

	// If the UTF16 string contains a small non-negative number, return that number.
	std::optional<uint32_t> parseGpuIndex( const std::wstring& requestedName )
	{
		if( requestedName.length() > 3 )
			return {};

		char buffer[ 4 ];
		*(uint32_t*)( &buffer[ 0 ] ) = 0;
		for( size_t i = 0; i < requestedName.length(); i++ )
		{
			const wchar_t wc = requestedName[ i ];
			if( wc < L'0' || wc > L'9' )
				return {};
			buffer[ i ] = (char)(uint8_t)wc;
		}

		uint32_t result;
		auto res = std::from_chars( buffer, &buffer[ 0 ] + requestedName.length(), result );
		if( res.ec == std::errc{} )
			return result;

		return {};
	}

	CComPtr<IDXGIAdapter1> selectAdapter( const std::wstring& requestedName )
	{
		if( requestedName.empty() )
			return nullptr;

		CComPtr<IDXGIFactory1> dxgi;
		HRESULT hr = createFactory( dxgi );
		if( FAILED( hr ) )
		{
			logWarningHr( hr, u8"CreateDXGIFactory1 failed" );
			return nullptr;
		}

		const auto idx = parseGpuIndex( requestedName );
		if( idx.has_value() )
		{
			// User has specified 0-based GPU index instead of the name
			// https://github.com/Const-me/Whisper/issues/72

			CComPtr<IDXGIAdapter1> adapter;
			hr = dxgi->EnumAdapters1( idx.value(), &adapter );
			if( hr == DXGI_ERROR_NOT_FOUND )
			{
				logWarning( u8"Requested GPU #%i not found", (int)idx.value() );
				return nullptr;
			}

			if( FAILED( hr ) )
			{
				logWarningHr( hr, u8"IDXGIFactory1.EnumAdapters1 failed" );
				return nullptr;
			}

			return adapter;
		}

		std::wstring name;
		for( UINT i = 0; true; i++ )
		{
			CComPtr<IDXGIAdapter1> adapter;
			hr = dxgi->EnumAdapters1( i, &adapter );
			if( hr == DXGI_ERROR_NOT_FOUND )
			{
				logWarning16( L"Requested GPU not found: \"%s\"", requestedName.c_str() );
				return nullptr;
			}

			if( FAILED( hr ) )
			{
				logWarningHr( hr, u8"IDXGIFactory1.EnumAdapters1 failed" );
				return nullptr;
			}

			DXGI_ADAPTER_DESC1 desc;
			adapter->GetDesc1( &desc );
			setName( name, desc );
			if( name == requestedName )
				return adapter;
		}
	}
}

HRESULT COMLIGHTCALL Whisper::listGPUs( pfnListAdapters pfn, void* pv )
{
	using namespace DirectCompute;

	CComPtr<IDXGIFactory1> dxgi;
	HRESULT hr = createFactory( dxgi );
	if( FAILED( hr ) )
	{
		logErrorHr( hr, u8"CreateDXGIFactory1 failed" );
		return hr;
	}

	std::wstring name;
	for( UINT i = 0; true; i++ )
	{
		CComPtr<IDXGIAdapter1> adapter;
		hr = dxgi->EnumAdapters1( i, &adapter );
		if( hr == DXGI_ERROR_NOT_FOUND )
			return S_OK;

		if( FAILED( hr ) )
		{
			logErrorHr( hr, u8"IDXGIFactory1.EnumAdapters1 failed" );
			return hr;
		}

		DXGI_ADAPTER_DESC1 desc;
		adapter->GetDesc1( &desc );
		setName( name, desc );
		pfn( name.c_str(), pv );
	}
}