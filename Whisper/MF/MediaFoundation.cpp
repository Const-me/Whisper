#include "stdafx.h"
#include "../API/iMediaFoundation.cl.h"
#include "mfStartup.h"
#include "../ComLightLib/comLightServer.h"
#include "loadAudioFile.h"
#include <mfidl.h>
#include <mfreadwrite.h>
#include "mfUtils.h"
#include "AudioCapture.h"
#include <mfapi.h>
#include <shlwapi.h>

namespace Whisper
{
	class AudioReader : public ComLight::ObjectRoot<iAudioReader>
	{
		CComPtr<IMFSourceReader> reader;
		bool wantStereo;
		CComPtr<iMediaFoundation> mediaFoundation;
		mutable int64_t preciseSamplesCount = 0;

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
			{
				if( 0 == preciseSamplesCount )
					return getStreamDuration( reader, rdi );
				else
				{
					rdi = MFllMulDiv( preciseSamplesCount, 10'000'000, SAMPLE_RATE, 0 );
					return S_OK;
				}
			}
			return OLE_E_BLANK;
		}
	public:

		HRESULT open( iMediaFoundation* owner, LPCTSTR path, bool stereo )
		{
			HRESULT hr = MFCreateSourceReaderFromURL( path, nullptr, &reader );
			if( FAILED( hr ) )
			{
				logError16( L"Unable to decode audio file \"%s\", MFCreateSourceReaderFromURL failed", path );
				return hr;
			}
			wantStereo = stereo;
			mediaFoundation = owner;
			logDebug16( L"Created source reader from the file \"%s\"", path );
			return S_OK;
		}

		HRESULT open( iMediaFoundation* owner, IMFByteStream* stream, bool stereo )
		{
			HRESULT hr = MFCreateSourceReaderFromByteStream( stream, nullptr, &reader );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"MFCreateSourceReaderFromByteStream failed" );
				return hr;
			}
			wantStereo = stereo;
			mediaFoundation = owner;
			logDebug( u8"Created source reader from the byte stream" );
			return S_OK;
		}
		void setPreciseSamplesCount( int64_t count ) const
		{
			preciseSamplesCount = count;
		}
	};

	void setPreciseSamplesCount( const iAudioReader* ar, int64_t count )
	{
		const AudioReader* r = static_cast<const AudioReader*>( ar );
		r->setPreciseSamplesCount( count );
	}

	class MediaFoundation : public ComLight::ObjectRoot<iMediaFoundation>
	{
		MfStartupRaii raii;
		DWORD tid = ~(DWORD)0;

		HRESULT COMLIGHTCALL loadAudioFile( LPCTSTR path, bool stereo, iAudioBuffer** pp ) const noexcept override final
		{
			return Whisper::loadAudioFile( path, stereo, pp );
		}
		HRESULT COMLIGHTCALL openAudioFile( LPCTSTR path, bool stereo, iAudioReader** pp ) noexcept override final
		{
			if( nullptr == path || nullptr == pp )
				return E_POINTER;

			ComLight::CComPtr<ComLight::Object<AudioReader>> res;
			CHECK( ComLight::Object<AudioReader>::create( res ) );
			CHECK( res->open( this, path, stereo ) );

			res.detach( pp );
			return S_OK;
		}
		HRESULT COMLIGHTCALL loadAudioFileData( const void* data, uint64_t size, bool stereo, iAudioReader** pp ) noexcept override final
		{
			if( nullptr == data || nullptr == pp )
				return E_POINTER;
			if( 0 != ( size >> 32 ) )
				return DISP_E_OVERFLOW;	// SHCreateMemStream is limited to 4GB, it seems

			CComPtr<IStream> comStream;
			// Microsoft neglected to document their API, but Wine returns a new stream with reference counter = 1
			// See shstream_create() function there https://source.winehq.org/source/dlls/shcore/main.c#0832
			// That's why we need the Attach(), as opposed to an assignment
			comStream.Attach( SHCreateMemStream( (const BYTE*)data, (UINT)size ) );
			if( !comStream )
			{
				logError( u8"SHCreateMemStream failed" );
				return E_FAIL;
			}

			CComPtr<IMFByteStream> mfStream;
			CHECK( MFCreateMFByteStreamOnStream( comStream, &mfStream ) );

			ComLight::CComPtr<ComLight::Object<AudioReader>> res;
			CHECK( ComLight::Object<AudioReader>::create( res ) );

			CHECK( res->open( this, mfStream, stereo ) );

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