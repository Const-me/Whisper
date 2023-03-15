#pragma once
#include "../../ComLightLib/comLightCommon.h"
#include "MfStructs.h"

struct IMFSourceReader;

namespace Whisper
{
	struct DECLSPEC_NOVTABLE iAudioBuffer : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "{013583aa-c9eb-42bc-83db-633c2c317051}" );

		virtual uint32_t COMLIGHTCALL countSamples() const = 0;
		virtual const float* COMLIGHTCALL getPcmMono() const = 0;
		virtual const float* COMLIGHTCALL getPcmStereo() const = 0;
		virtual HRESULT COMLIGHTCALL getTime( int64_t& rdi ) const = 0;
	};

	struct DECLSPEC_NOVTABLE iAudioReader : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "{35b988da-04a6-476a-a193-d8891d5dc390}" );

		virtual HRESULT COMLIGHTCALL getDuration( int64_t& rdi ) const = 0;
		virtual HRESULT COMLIGHTCALL getReader( IMFSourceReader** pp ) const = 0;
		virtual HRESULT COMLIGHTCALL requestedStereo() const = 0;
	};

	struct DECLSPEC_NOVTABLE iAudioCapture : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "{747752c2-d9fd-40df-8847-583c781bf013}" );

		virtual HRESULT COMLIGHTCALL getReader( IMFSourceReader** pp ) const = 0;
		virtual const sCaptureParams& COMLIGHTCALL getParams() const = 0;
	};

	struct DECLSPEC_NOVTABLE iMediaFoundation : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "{fb9763a5-d77d-4b6e-aff8-f494813cebd8}" );

		virtual HRESULT COMLIGHTCALL loadAudioFile( LPCTSTR path, bool stereo, iAudioBuffer** pp ) const = 0;
		virtual HRESULT COMLIGHTCALL openAudioFile( LPCTSTR path, bool stereo, iAudioReader** pp ) = 0;
		virtual HRESULT COMLIGHTCALL loadAudioFileData( const void* data, uint64_t size, bool stereo, iAudioReader** pp ) = 0;

		virtual HRESULT COMLIGHTCALL listCaptureDevices( pfnFoundCaptureDevices pfn, void* pv ) = 0;
		virtual HRESULT COMLIGHTCALL openCaptureDevice( LPCTSTR endpoint, const sCaptureParams& captureParams, iAudioCapture** pp ) = 0;
	};

	HRESULT COMLIGHTCALL initMediaFoundation( iMediaFoundation** pp );
}