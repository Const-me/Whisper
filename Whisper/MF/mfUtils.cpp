#include "stdafx.h"
#include "mfUtils.h"
#include <mfapi.h>

HRESULT Whisper::createMediaType( bool stereo, IMFMediaType** pp )
{
	if( nullptr == pp )
		return E_POINTER;

	CComPtr<IMFMediaType> mt;
	CHECK( MFCreateMediaType( &mt ) );
	CHECK( mt->SetGUID( MF_MT_MAJOR_TYPE, MFMediaType_Audio ) );
	CHECK( mt->SetGUID( MF_MT_SUBTYPE, MFAudioFormat_Float ) );
	CHECK( mt->SetUINT32( MF_MT_AUDIO_SAMPLES_PER_SECOND, SAMPLE_RATE ) );

	const uint32_t channels = stereo ? 2 : 1;
	CHECK( mt->SetUINT32( MF_MT_AUDIO_NUM_CHANNELS, channels ) );
	CHECK( mt->SetUINT32( MF_MT_AUDIO_BLOCK_ALIGNMENT, channels * 4 ) );
	CHECK( mt->SetUINT32( MF_MT_AUDIO_AVG_BYTES_PER_SECOND, channels * 4 * SAMPLE_RATE ) );
	CHECK( mt->SetUINT32( MF_MT_AUDIO_BITS_PER_SAMPLE, 32 ) );
	CHECK( mt->SetUINT32( MF_MT_ALL_SAMPLES_INDEPENDENT, TRUE ) );

	*pp = mt.Detach();

	return S_OK;
}

HRESULT Whisper::getStreamDuration( IMFSourceReader* reader, int64_t& duration )
{
	PROPVARIANT var;
	PropVariantInit( &var );
	CHECK( reader->GetPresentationAttribute( MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &var ) );

	if( var.vt == VT_UI8 )
	{
		// The documentation says the type of that attribute is UINT64
		// https://learn.microsoft.com/en-us/windows/win32/medfound/mf-pd-duration-attribute
		duration = var.uhVal.QuadPart;
		return S_OK;
	}
	logError( u8"Unexpected type of MF_PD_DURATION attribute" );
	return E_INVALIDARG;
}

HRESULT Whisper::validateCurrentMediaType( IMFSourceReader* reader, uint32_t expectedChannels )
{
	CComPtr<IMFMediaType> mt;
	CHECK( reader->GetCurrentMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, &mt ) );

	GUID guid;
	CHECK( mt->GetGUID( MF_MT_MAJOR_TYPE, &guid ) );
	if( guid != MFMediaType_Audio )
		return E_FAIL;

	CHECK( mt->GetGUID( MF_MT_SUBTYPE, &guid ) );
	if( guid != MFAudioFormat_Float )
		return E_FAIL;

	UINT32 u32;
	CHECK( mt->GetUINT32( MF_MT_AUDIO_SAMPLES_PER_SECOND, &u32 ) );
	if( u32 != SAMPLE_RATE )
		return E_FAIL;

	CHECK( mt->GetUINT32( MF_MT_AUDIO_NUM_CHANNELS, &u32 ) );
	if( u32 != expectedChannels )
		return E_FAIL;

	return S_OK;
}