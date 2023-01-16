#include "stdafx.h"
#include "PcmReader.h"
#include <mfapi.h>
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
				rdi.mono.resize( block * 2 );
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
}

PcmReader::PcmReader( IMFSourceReader* reader, bool stereo )
{
	if( nullptr == reader )
		throw E_POINTER;
	this->reader = reader;

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

	// Find out the length
	int64_t durationTicks;
	check( getStreamDuration( reader, durationTicks ) );

	// Convert length to chunks
	// Seconds = Ticks / 10^7
	// Samples = Seconds * SAMPLE_RATE = Ticks * SAMPLE_RATE / 10^7
	// Chunks = Samples / FFT_STEP = Ticks * SAMPLE_RATE / ( FFT_STEP * 10^7 ), and we want that integer rounded down
	constexpr __int64 mul = SAMPLE_RATE;
	constexpr __int64 div = (__int64)FFT_STEP * 10'000'000;
	m_length = (size_t)MFllMulDiv( durationTicks, mul, div, 0 );
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