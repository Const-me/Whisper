#include "stdafx.h"
#include "listGPUs.h"
#pragma comment(lib, "DXGI.lib")

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
				logErrorHr( hr, u8"IDXGIFactory1.EnumAdapters1 failed" );
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