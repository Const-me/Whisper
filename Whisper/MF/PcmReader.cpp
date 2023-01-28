#include "stdafx.h"
#include "PcmReader.h"
#include <mfapi.h>
#include <Mferror.h>
#include "mfUtils.h"

namespace Whisper
{
	__interface iSampleHandler
	{
		void copyChunk( PcmMonoChunk* pMono, const AudioBuffer& rsi, size_t sourceOffset, PcmStereoChunk* pStereo ) const;
		void moveBufferData( AudioBuffer& rdi, size_t amount ) const;
		void appendPcm( AudioBuffer& rdi, const float* rsi, size_t countFloats ) const;
		void copyChunk( PcmMonoChunk* pMono, const AudioBuffer& rsi, size_t sourceOffset, size_t samples, PcmStereoChunk* pStereo ) const;
		uint32_t readerChannelsCount() const;
	};
}

namespace
{
	using namespace Whisper;

	__forceinline void copyMono( PcmMonoChunk* rdi, const AudioBuffer& rsi, size_t sourceOffset, size_t samples )
	{
		assert( sourceOffset + samples <= rsi.mono.size() );
		memcpy( rdi->mono.data(), &rsi.mono[ sourceOffset ], samples * 4 );
		if( samples < FFT_STEP )
			memset( rdi->mono.data() + samples, 0, ( FFT_STEP - samples ) * 4 );
	}

	__forceinline void copyStereo( PcmStereoChunk* rdi, const AudioBuffer& rsi, size_t sourceOffset, size_t samples )
	{
		memcpy( rdi->stereo.data(), &rsi.stereo[ sourceOffset * 2 ], samples * 8 );
		if( samples < FFT_STEP )
			memset( rdi->stereo.data() + samples * 2, 0, ( FFT_STEP - samples ) * 8 );
	}

	struct HandlerMono : iSampleHandler
	{
		void appendPcm( AudioBuffer& rdi, const float* rsi, size_t countFloats ) const override
		{
			rdi.appendMono( rsi, countFloats );
		}
		void copyChunk( PcmMonoChunk* pMono, const AudioBuffer& rsi, size_t sourceOffset, PcmStereoChunk* pStereo ) const override final
		{
			copyMono( pMono, rsi, sourceOffset, FFT_STEP );
		}
		void copyChunk( PcmMonoChunk* pMono, const AudioBuffer& rsi, size_t sourceOffset, size_t samples, PcmStereoChunk* pStereo ) const override final
		{
			copyMono( pMono, rsi, sourceOffset, samples );
		}
		void moveBufferData( AudioBuffer& rdi, size_t amount ) const override final
		{
			const size_t len = rdi.mono.size();
			assert( amount <= len );
			if( amount < len )
			{
				const size_t block = len - amount;
				memmove( rdi.mono.data(), rdi.mono.data() + amount, block * 4 );
				rdi.mono.resize( block );
			}
			else
				rdi.mono.clear();
		}
		uint32_t readerChannelsCount() const override { return 1; }
	};
	struct HandlerDownmixedStereo : HandlerMono
	{
		void appendPcm( AudioBuffer& rdi, const float* rsi, size_t countFloats ) const override final
		{
			rdi.appendDownmixedStereo( rsi, countFloats );
		}
		uint32_t readerChannelsCount() const override final { return 2; }
	};
	struct HandlerStereo : iSampleHandler
	{
		void appendPcm( AudioBuffer& rdi, const float* rsi, size_t countFloats ) const override final
		{
			rdi.appendStereo( rsi, countFloats );
		}
		void copyChunk( PcmMonoChunk* pMono, const AudioBuffer& rsi, size_t sourceOffset, PcmStereoChunk* pStereo ) const override final
		{
			copyMono( pMono, rsi, sourceOffset, FFT_STEP );
			copyStereo( pStereo, rsi, sourceOffset, FFT_STEP );
		}
		void copyChunk( PcmMonoChunk* pMono, const AudioBuffer& rsi, size_t sourceOffset, size_t samples, PcmStereoChunk* pStereo ) const override final
		{
			copyMono( pMono, rsi, sourceOffset, samples );
			copyStereo( pStereo, rsi, sourceOffset, samples );
		}
		void moveBufferData( AudioBuffer& rdi, size_t amount ) const override final
		{
			const size_t len = rdi.mono.size();
			assert( amount <= len );
			if( amount < len )
			{
				const size_t block = len - amount;
				memmove( rdi.mono.data(), rdi.mono.data() + amount, block * 4 );
				rdi.mono.resize( block );
				memmove( rdi.stereo.data(), rdi.stereo.data() + amount * 2, block * 8 );
				rdi.stereo.resize( block * 2 );
			}
			else
			{
				rdi.mono.clear();
				rdi.stereo.clear();
			}
		}
		uint32_t readerChannelsCount() const override final { return 2; }
	};
	static const HandlerMono s_mono;
	static const HandlerDownmixedStereo s_downmix;
	static const HandlerStereo s_stereo;

	__forceinline __m128i load( const GUID& guid )
	{
		return _mm_loadu_si128( ( const __m128i* )( &guid ) );
	}

	// Find audio decoder MFT, query MF_MT_SUBTYPE attribute of the current input media type of that MFT
	HRESULT getDecoderInputSubtype( IMFSourceReader* reader, __m128i& rdi )
	{
		store16( &rdi, _mm_setzero_si128() );

		CComPtr<IMFSourceReaderEx> readerEx;
		CHECK( reader->QueryInterface( &readerEx ) );
		constexpr uint32_t stream = MF_SOURCE_READER_FIRST_AUDIO_STREAM;
		const __m128i decGuid = load( MFT_CATEGORY_AUDIO_DECODER );
		alignas( 16 ) GUID category;
		for( DWORD i = 0; true; i++ )
		{
			CComPtr<IMFTransform> mft;
			HRESULT hr = readerEx->GetTransformForStream( stream, i, &category, &mft );
			if( hr == MF_E_INVALIDINDEX )
			{
				// This happens for *.wav input files
				// They don't have any MFT_CATEGORY_AUDIO_DECODER MFTs in the source reader, and it's not an error
				return S_FALSE;
			}
			if( FAILED( hr ) )
				return hr;
			const __m128i cat = _mm_load_si128( ( const __m128i* ) & category );
			if( !vectorEqual( decGuid, cat ) )
				continue;

			CComPtr<IMFMediaType> mt;
			CHECK( mft->GetInputCurrentType( 0, &mt ) );
			CHECK( mt->GetGUID( MF_MT_SUBTYPE, (GUID*)&rdi ) );
			return S_OK;
		}
	}

	// S_OK when the reader has an MP3 decoder for the first audio stream, S_FALSE otherwise
	HRESULT isMp3Decoder( IMFSourceReader* reader )
	{
		__m128i subtype;
		CHECK( getDecoderInputSubtype( reader, subtype ) );
		const bool res = vectorEqual( subtype, load( MFAudioFormat_MP3 ) );
		return res ? S_OK : S_FALSE;
	}

	// Workaround for a Microsoft's bug in Media Foundation MP3 decoder: https://github.com/Const-me/Whisper/issues/4
	// Media Foundation is reporting incorrect media duration = 12.54. Windows Media Player does the same.
	// Winamp and Media Player Classic are reporting 12:35, VLC reports 12:36.
	HRESULT getPreciseDuration( IMFSourceReader* reader, size_t& length, bool mono, const iAudioReader* iar )
	{
		size_t samples = 0;

		// Decode the complete stream, counting samples
		while( true )
		{
			DWORD dwFlags = 0;
			CComPtr<IMFSample> sample;

			// Read the next sample
			HRESULT hr = reader->ReadSample( (DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &dwFlags, nullptr, &sample );
			if( FAILED( hr ) )
			{
				logErrorHr( hr, u8"IMFSourceReader.ReadSample" );
				return hr;
			}

			if( dwFlags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED )
			{
				// logError( u8"Media type changes ain’t supported by the library." );
				// return E_UNEXPECTED;

				// This happens for some video files at the very start of the reading, with Dolby AC3 audio track.
				// Instead of failing the transcribe process, verify the important attributes (FP32 samples, sample rate, count of channels) haven’t changed.
				CHECK( validateCurrentMediaType( reader, mono ? 1 : 2 ) );
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

			assert( 0 == ( cbBuffer % sizeof( float ) ) );
			const size_t countFloats = cbBuffer / sizeof( float );
			if( mono )
				samples += countFloats;
			else
			{
				assert( 0 == countFloats % 2 );
				samples += countFloats / 2;
			}

			// Unlock the buffer
			hr = buffer->Unlock();
			if( FAILED( hr ) )
				return hr;
		}

		// Rewind the stream to beginning
		PROPVARIANT pv;
		PropVariantInit( &pv );
		pv.vt = VT_I8;
		pv.hVal.QuadPart = 0;
		CHECK( reader->SetCurrentPosition( GUID_NULL, pv ) );

		// Make the output value
		length = samples / FFT_STEP;

		// Store the actual samples count in the reader
		// This way the iAudioReader.getDuration() API returns correct value to the user
		setPreciseSamplesCount( iar, samples );

		return S_OK;
	}

	HRESULT getDuration( IMFSourceReader* reader, size_t& length, bool mono, const iAudioReader* iar )
	{
		HRESULT hr = isMp3Decoder( reader );
		if( SUCCEEDED( hr ) )
		{
			if( S_OK == hr )
			{
				return getPreciseDuration( reader, length, mono, iar );
			}
		}
		else
			logWarningHr( hr, u8"isMp3Decoder" );

		// Find out the length
		int64_t durationTicks;
		CHECK( getStreamDuration( reader, durationTicks ) );

		// Convert length to chunks
		// Seconds = Ticks / 10^7
		// Samples = Seconds * SAMPLE_RATE = Ticks * SAMPLE_RATE / 10^7
		// Chunks = Samples / FFT_STEP = Ticks * SAMPLE_RATE / ( FFT_STEP * 10^7 ), and we want that integer rounded down
		constexpr __int64 mul = SAMPLE_RATE;
		constexpr __int64 div = (__int64)FFT_STEP * 10'000'000;
		length = (size_t)MFllMulDiv( durationTicks, mul, div, 0 );
		return S_OK;
	}
}

PcmReader::PcmReader( const iAudioReader* iar )
{
	if( nullptr == iar )
		throw E_POINTER;

	check( iar->getReader( &reader ) );
	const bool stereo = iar->requestedStereo() == S_OK;

	// Set up media type, and figure out sample handler
	check( reader->SetStreamSelection( MF_SOURCE_READER_ALL_STREAMS, FALSE ) );
	check( reader->SetStreamSelection( MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE ) );

	CComPtr<IMFMediaType> mtNative;
	check( reader->GetNativeMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_SOURCE_READER_CURRENT_TYPE_INDEX, &mtNative ) );
	UINT32 numChannels;
	check( mtNative->GetUINT32( MF_MT_AUDIO_NUM_CHANNELS, &numChannels ) );

	const bool sourceMono = numChannels < 2;
	if( sourceMono )
		sampleHandler = &s_mono;
	else if( !stereo )
		sampleHandler = &s_downmix;
	else
	{
		sampleHandler = &s_stereo;
		m_stereoOutput = true;
	}

	CComPtr<IMFMediaType> mt;
	check( createMediaType( !sourceMono, &mt ) );
	check( reader->SetCurrentMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, mt ) );

	// Find out the length.
	// Sadly, broken Microsoft's MP3 decoder MFT made this much harder than necessary:
	// https://github.com/Const-me/Whisper/issues/4
	check( getDuration( reader, m_length, sourceMono, iar ) );
}

HRESULT PcmReader::readNextSample()
{
	const size_t off = bufferReadOffset;
	const size_t availableSamples = pcm.mono.size() - off;

	// If needed, move the remaining PCM data to the start of these vectors
	if( availableSamples > 0 )
	{
		if( 0 != off )
			sampleHandler->moveBufferData( pcm, off );
	}
	else
		pcm.clear();
	bufferReadOffset = 0;

	while( true )
	{
		DWORD dwFlags = 0;
		CComPtr<IMFSample> sample;

		// Read the next sample
		HRESULT hr = reader->ReadSample( (DWORD)MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0, nullptr, &dwFlags, nullptr, &sample );
		if( FAILED( hr ) )
		{
			logErrorHr( hr, u8"IMFSourceReader.ReadSample" );
			return hr;
		}

		if( dwFlags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED )
		{
			// logError( u8"Media type changes ain’t supported by the library." );
			// return E_UNEXPECTED;

			// This happens for some video files at the very start of the reading, with Dolby AC3 audio track.
			// Instead of failing the transcribe process, verify the important attributes (FP32 samples, sample rate, count of channels) haven’t changed.
			CHECK( validateCurrentMediaType( reader, sampleHandler->readerChannelsCount() ) );
		}

		if( dwFlags & MF_SOURCE_READERF_ENDOFSTREAM )
			return E_EOF;

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
			assert( 0 == ( cbBuffer % sizeof( float ) ) );
			const size_t countFloats = cbBuffer / sizeof( float );
			sampleHandler->appendPcm( pcm, pAudioData, countFloats );
		}
		catch( const std::bad_alloc& )
		{
			buffer->Unlock();
			return E_OUTOFMEMORY;
		}

		// Unlock the buffer
		hr = buffer->Unlock();
		if( FAILED( hr ) )
			return hr;

		return S_OK;
	}
}

HRESULT PcmReader::readChunk( PcmMonoChunk& mono, PcmStereoChunk* stereo )
{
	while( true )
	{
		const size_t off = bufferReadOffset;
		const size_t availableSamples = pcm.mono.size() - off;
		if( availableSamples >= FFT_STEP )
		{
			// We have enough data in the buffer
			sampleHandler->copyChunk( &mono, pcm, off, stereo );
			bufferReadOffset = off + FFT_STEP;
			return S_OK;
		}

		if( !m_readerEndOfFile )
		{
			// We don't have enough data, but the stream has not ended yet, can load moar samples from the reader
			HRESULT hr = readNextSample();
			if( SUCCEEDED( hr ) )
				continue;
			if( hr != E_EOF )
				return hr;
			m_readerEndOfFile = true;
		}

		if( availableSamples > 0 )
		{
			// We have reached the end of stream of the reader, but the buffer still has a few samples.
			// Return the final incomplete chunk padded with zeros
			sampleHandler->copyChunk( &mono, pcm, off, availableSamples, stereo );
			bufferReadOffset = off + availableSamples;
			return S_OK;
		}

		return E_EOF;
	}
}