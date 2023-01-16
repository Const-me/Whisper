#include "stdafx.h"
#include <atlstr.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include "AudioCapture.h"
#include "../API/iMediaFoundation.cl.h"
#include "../ComLightLib/comLightServer.h"
#pragma comment(lib, "Mf.lib")

namespace
{
	struct Strings
	{
		CString displayName, endpoint;
	};

	HRESULT getAllocString( IMFActivate* activate, const GUID& id, CString& rdi )
	{
		wchar_t* pointer = nullptr;
		UINT32 cchName;
		HRESULT hr = activate->GetAllocatedString( id, &pointer, &cchName );
		if( SUCCEEDED( hr ) )
			rdi.SetString( pointer, cchName );
		CoTaskMemFree( pointer );
		return hr;
	}

	HRESULT getInfo( IMFActivate* activate, Strings& rdi )
	{
		CHECK( getAllocString( activate, MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, rdi.displayName ) );
		CHECK( getAllocString( activate, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_AUDCAP_ENDPOINT_ID, rdi.endpoint ) );
		return S_OK;
	}

	HRESULT __stdcall supplyDevices( Whisper::pfnFoundCaptureDevices pfn, void* pv, IMFActivate** ppDevices, UINT32 count )
	{
		if( ppDevices == nullptr || count == 0 )
			return pfn( 0, nullptr, pv );

		std::vector<Strings> strings;
		strings.reserve( count );

		for( UINT i = 0; i < count; i++ )
		{
			IMFActivate* const activate = ppDevices[ i ];
			if( nullptr == activate )
				continue;
			Strings info;
			HRESULT hr = getInfo( activate, info );
			if( FAILED( hr ) )
				continue;

			strings.emplace_back( std::move( info ) );
		}

		const size_t len = strings.size();
		if( 0 == len )
			return pfn( 0, nullptr, pv );

		std::vector<Whisper::sCaptureDevice> pointers;
		pointers.resize( len );
		for( size_t i = 0; i < len; i++ )
		{
			const auto& src = strings[ i ];
			auto& dest = pointers[ i ];
			dest.displayName = src.displayName;
			dest.endpoint = src.endpoint;
		}
		return pfn( (int)len, pointers.data(), pv );
	}
}

HRESULT __stdcall Whisper::captureDeviceList( pfnFoundCaptureDevices pfn, void* pv )
{
	// Create an attribute store to hold the search criteria.
	CComPtr<IMFAttributes> attrs;
	CHECK( MFCreateAttributes( &attrs, 1 ) );
	// Request audio capture devices
	CHECK( attrs->SetGUID( MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_AUDCAP_GUID ) );

	// Enumerate the devices
	IMFActivate** ppDevices = nullptr;
	UINT32 count = 0;
	CHECK( MFEnumDeviceSources( attrs, &ppDevices, &count ) );

	// Feed the data to the caller
	HRESULT hr = supplyDevices( pfn, pv, ppDevices, count );

	// Free the memory
	for( DWORD i = 0; i < count; i++ )
		ppDevices[ i ]->Release();
	CoTaskMemFree( ppDevices );

	return hr;
}

namespace
{
	using namespace Whisper;

	class Capture : public ComLight::ObjectRoot<iAudioCapture>
	{
		CComPtr<IMFSourceReader> reader;
		CComPtr<iMediaFoundation> mediaFoundation;
		sCaptureParams captureParams;

		HRESULT COMLIGHTCALL getReader( IMFSourceReader** pp ) const noexcept override final
		{
			if( pp == nullptr )
				return E_POINTER;
			CComPtr<IMFSourceReader> res = reader;
			*pp = res.Detach();;
			return S_OK;
		}
		const sCaptureParams& COMLIGHTCALL getParams() const noexcept override final
		{
			return captureParams;
		}
	public:
		HRESULT open( iMediaFoundation* owner, const wchar_t* endpoint, const sCaptureParams& cp );
	};

	HRESULT Capture::open( iMediaFoundation* owner, const wchar_t* endpoint, const sCaptureParams& cp )
	{
		// Create an attribute store to hold the search criteria.
		CComPtr<IMFAttributes> attrs;
		CHECK( MFCreateAttributes( &attrs, 2 ) );
		// Request audio capture devices
		CHECK( attrs->SetGUID( MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_AUDCAP_GUID ) );
		CHECK( attrs->SetString( MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_AUDCAP_ENDPOINT_ID, endpoint ) );

		CComPtr<IMFMediaSource> source;
		HRESULT hr = MFCreateDeviceSource( attrs, &source );
		if( FAILED( hr ) )
		{
			logErrorHr( hr, u8"MFCreateDeviceSource" );
			return hr;
		}

		// TODO: implement IMFSourceReaderCallback, pass into MF_SOURCE_READER_ASYNC_CALLBACK attribute
		// This is to support cancellation
		hr = MFCreateSourceReaderFromMediaSource( source, nullptr, &reader );
		if( FAILED( hr ) )
		{
			logErrorHr( hr, u8"MFCreateSourceReaderFromMediaSource" );
			return hr;
		}

		captureParams = cp;
		mediaFoundation = owner;
		return S_OK;
	}
}

HRESULT __stdcall Whisper::captureOpen( iMediaFoundation* owner, const wchar_t* endpoint, const sCaptureParams& captureParams, iAudioCapture** pp ) noexcept
{
	if( nullptr == endpoint || nullptr == pp )
		return E_POINTER;

	ComLight::CComPtr<ComLight::Object<Capture>> res;
	CHECK( ComLight::Object<Capture>::create( res ) );
	CHECK( res->open( owner, endpoint, captureParams ) );

	res.detach( pp );
	return S_OK;
}