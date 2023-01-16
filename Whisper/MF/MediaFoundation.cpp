#include "stdafx.h"
#include "../API/iMediaFoundation.cl.h"
#include "mfStartup.h"
#include "../ComLightLib/comLightServer.h"
#include "loadAudioFile.h"
#include <mfidl.h>
#include <mfreadwrite.h>
#include "mfUtils.h"
#include "AudioCapture.h"

namespace Whisper
{
	class AudioReader : public ComLight::ObjectRoot<iAudioReader>
	{
		CComPtr<IMFSourceReader> reader;
		bool wantStereo;
		CComPtr<iMediaFoundation> mediaFoundation;

		HRESULT COMLIGHTCALL getReader( IMFSourceReader** pp ) const noexcept override final
		{
			if( pp == nullptr )
				return E_POINTER;
			CComPtr<IMFSourceReader> res = reader;
			*pp = res.Detach();;
			return S_OK;
		}
		HRESULT COMLIGHTCALL requestedStereo() const noexcept override final
		{
			return wantStereo ? S_OK : S_FALSE;
		}
		HRESULT COMLIGHTCALL getDuration( int64_t& rdi ) const noexcept override final
		{
			if( reader )
				return getStreamDuration( reader, rdi );
			return OLE_E_BLANK;
		}
	public:
		HRESULT open( iMediaFoundation* owner, LPCTSTR path, bool stereo )
		{
			HRESULT hr = MFCreateSourceReaderFromURL( path, nullptr, &reader );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"MFCreateSourceReaderFromURL failed" );
				return hr;
			}
			wantStereo = stereo;
			mediaFoundation = owner;
			logDebug16( L"Created source reader from the file \"%s\"", path );
			return S_OK;
		}
	};

	class MediaFoundation : public ComLight::ObjectRoot<iMediaFoundation>
	{
		MfStartupRaii raii;
		DWORD tid = ~(DWORD)0;

		virtual HRESULT COMLIGHTCALL loadAudioFile( LPCTSTR path, bool stereo, iAudioBuffer** pp ) const noexcept override final
		{
			return Whisper::loadAudioFile( path, stereo, pp );
		}
		virtual HRESULT COMLIGHTCALL openAudioFile( LPCTSTR path, bool stereo, iAudioReader** pp ) noexcept override final
		{
			if( nullptr == path || nullptr == pp )
				return E_POINTER;

			ComLight::CComPtr<ComLight::Object<AudioReader>> res;
			CHECK( ComLight::Object<AudioReader>::create( res ) );
			CHECK( res->open( this, path, stereo ) );

			res.detach( pp );
			return S_OK;
		}
		HRESULT COMLIGHTCALL listCaptureDevices( pfnFoundCaptureDevices pfn, void* pv ) noexcept override final
		{
			return captureDeviceList( pfn, pv );
		}
		HRESULT COMLIGHTCALL openCaptureDevice( LPCTSTR endpoint, const sCaptureParams& captureParams, iAudioCapture** pp ) noexcept override final
		{
			return captureOpen( this, endpoint, captureParams, pp );
		}
	protected:

		HRESULT FinalConstruct()
		{
			CHECK( raii.startup() );
			tid = GetCurrentThreadId();
			return S_OK;
		}

	public:

		~MediaFoundation() override
		{
			assert( tid == GetCurrentThreadId() );
		}
	};
}

HRESULT COMLIGHTCALL Whisper::initMediaFoundation( iMediaFoundation** pp )
{
	if( nullptr == pp )
		return E_POINTER;

	ComLight::CComPtr<ComLight::Object<MediaFoundation>> obj;
	CHECK( ComLight::Object<MediaFoundation>::create( obj ) );
	obj.detach( pp );
	return S_OK;
}