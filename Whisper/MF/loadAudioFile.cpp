#include "stdafx.h"
#include "../ComLightLib/comLightServer.h"
#include "loadAudioFile.h"
#include "mfUtils.h"
#include "AudioBuffer.h"
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfapi.h>
#include <shlwapi.h>
#pragma comment(lib, "Mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

namespace Whisper
{
	class MediaFileBuffer : public ComLight::ObjectRoot<iAudioBuffer>
	{
		AudioBuffer pcm;
		uint32_t channels = 0;

		uint32_t COMLIGHTCALL countSamples() const noexcept override final
		{
			return (uint32_t)( pcm.mono.size() );
		}
		const float* COMLIGHTCALL getPcmMono() const noexcept override final
		{
			if( !pcm.mono.empty() )
				return pcm.mono.data();
			return nullptr;
		}
		const float* COMLIGHTCALL getPcmStereo() const noexcept override final
		{
			if( !pcm.stereo.empty() )
				return pcm.stereo.data();
			return nullptr;
		}
		HRESULT COMLIGHTCALL getTime( int64_t& rdi ) const noexcept override final
		{
			rdi = 0;
			return S_OK;
		}
		HRESULT _loadBuffer(CComPtr<IMFSourceReader> reader, LPCTSTR path, bool stereo);
	public:
		HRESULT load( LPCTSTR path, bool stereo );
		HRESULT loadBuffer(const void* data, uint64_t size, bool stereo);
	};

	HRESULT MediaFileBuffer::load( LPCTSTR path, bool stereo )
	{
		CComPtr<IMFSourceReader> reader;
		HRESULT hr = MFCreateSourceReaderFromURL( path, nullptr, &reader );
		if( FAILED( hr ) )
		{
			logError16( L"Unable to decode audio file \"%s\", MFCreateSourceReaderFromURL failed", path );
			return hr;
		}

		return _loadBuffer(reader, path, stereo);
	}


	HRESULT MediaFileBuffer::_loadBuffer(CComPtr<IMFSourceReader> reader, LPCTSTR path, bool stereo)
	{

		CHECK(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE));
		CHECK(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE));


		CComPtr<IMFMediaType> mtNative;
		CHECK( reader->GetNativeMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_SOURCE_READER_CURRENT_TYPE_INDEX, &mtNative ) );
		UINT32 numChannels;
		CHECK( mtNative->GetUINT32( MF_MT_AUDIO_NUM_CHANNELS, &numChannels ) );
		const bool sourceMono = numChannels == 1;
		const AudioBuffer::pfnAppendSamples pfn = AudioBuffer::appendSamplesFunc( sourceMono, stereo );
		channels = ( stereo && !sourceMono ) ? 2 : 1;

		CComPtr<IMFMediaType> mt;
		CHECK( createMediaType( !sourceMono, &mt ) );

		CHECK( reader->SetCurrentMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, mt ) );

		while( true )
		{
			DWORD dwFlags = 0;
			CComPtr<IMFSample> sample;

			// Read the next sample.
			hr = reader->ReadSample( (DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &dwFlags, nullptr, &sample );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"IMFSourceReader.ReadSample" );
				return hr;
			}

			if( dwFlags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED )
			{
				logError( u8"Media type changes ain’t supported by the library." );
				return E_UNEXPECTED;
			}

			if( dwFlags & MF_SOURCE_READERF_ENDOFSTREAM )
				break;

			if( !sample )
			{
				// printf( "No sample\n" );
				continue;
			}

			// Get a pointer to the audio data in the sample.
			CComPtr<IMFMediaBuffer> buffer;
			hr = sample->ConvertToContiguousBuffer( &buffer );
			if( FAILED( hr ) )
				return hr;

			const float* pAudioData = nullptr;
			DWORD cbBuffer;
			hr = buffer->Lock( (BYTE**)&pAudioData, nullptr, &cbBuffer );
			if( FAILED( hr ) )
				return hr;

			try
			{
				const size_t countFloats = cbBuffer / sizeof( float );
				( pcm.*pfn )( pAudioData, countFloats );
			}
			catch( const std::bad_alloc& )
			{
				return E_OUTOFMEMORY;
			}

			// Unlock the buffer
			hr = buffer->Unlock();
			if( FAILED( hr ) )
				return hr;
		}

		const size_t len = pcm.mono.size();
		if( len == 0 )
		{
			logError16( L"The audio file \"%s\" has no samples", path );
			return E_INVALIDARG;
		}
		if( len < SAMPLE_RATE / 2 )
			logError16( L"The file \"%s\" only has %zu samples, less than 0.5 seconds of audio", path, len );
		else
			logDebug16( L"Loaded audio file from \"%s\": %zu samples, %g seconds", path, len, (int)len * ( 1.0 / SAMPLE_RATE ) );
		return S_OK;

	}

	

	HRESULT MediaFileBuffer::loadBuffer(const void* data, uint64_t size, bool stereo)
	{

		CComPtr<IMFSourceReader> reader;
		CComPtr<IStream> comStream;
		// Microsoft neglected to document their API, but Wine returns a new stream with reference counter = 1
		// See shstream_create() function there https://source.winehq.org/source/dlls/shcore/main.c#0832
		// That's why we need the Attach(), as opposed to an assignment
		comStream.Attach(SHCreateMemStream((const BYTE*)data, (UINT)size));
		if (!comStream)
		{
			logError(u8"SHCreateMemStream failed");
			return E_FAIL;
		}

		CComPtr<IMFByteStream> mfStream;
		CHECK(MFCreateMFByteStreamOnStream(comStream, &mfStream));
		HRESULT hr = MFCreateSourceReaderFromByteStream(mfStream, nullptr, &reader);

		return _loadBuffer(reader, L"memory", stereo);
	}
}



HRESULT COMLIGHTCALL Whisper::loadAudioFile( LPCTSTR path, bool stereo, iAudioBuffer** pp )
{
	if( nullptr == path || nullptr == pp )
		return E_POINTER;

	ComLight::CComPtr<ComLight::Object<MediaFileBuffer>> obj;
	CHECK( ComLight::Object<MediaFileBuffer>::create( obj ) );
	CHECK( obj->load( path, stereo ) );
	obj.detach( pp );
	return S_OK;
}

HRESULT COMLIGHTCALL Whisper::loadAudioMemoryFile(const void *data, uint64_t size, bool stereo, iAudioBuffer** pp)
{
	if (nullptr == data || nullptr == pp || size == 0)
		return E_POINTER;

	ComLight::CComPtr<ComLight::Object<MediaFileBuffer>> obj;
	CHECK(ComLight::Object<MediaFileBuffer>::create(obj));
	CHECK(obj->loadBuffer(data, size, stereo));
	obj.detach(pp); 
	return S_OK;
}