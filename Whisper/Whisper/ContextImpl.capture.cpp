#include "stdafx.h"
#include "ContextImpl.h"
#include "../API/iMediaFoundation.cl.h"
#include "../MF/AudioBuffer.h"
#include "../MF/mfUtils.h"
#include <mfidl.h>
#include <mfapi.h>
#include <mfreadwrite.h>
#include "voiceActivityDetection.h"

namespace
{
	using namespace Whisper;

	class TranscribeBuffer : public ComLight::ObjectRoot<iAudioBuffer>
	{
		// ==== iAudioBuffer ====
		uint32_t COMLIGHTCALL countSamples() const override final
		{
			return (uint32_t)pcm.mono.size();
		}
		const float* COMLIGHTCALL getPcmMono() const override final
		{
			if( !pcm.mono.empty() )
				return pcm.mono.data();
			return nullptr;
		}
		const float* COMLIGHTCALL getPcmStereo() const override final
		{
			if( !pcm.stereo.empty() )
				return pcm.stereo.data();
			return nullptr;
		}
		HRESULT COMLIGHTCALL getTime( int64_t& rdi ) const override final
		{
			rdi = MFllMulDiv( currentOffset, 10'000'000, SAMPLE_RATE, 0 );
			return S_OK;
		}
	public:
		AudioBuffer pcm;
		int64_t currentOffset = 0;
	};

	class TranscribeBufferObj : public ComLight::Object<TranscribeBuffer>
	{
		uint32_t Release() override final
		{
			return RefCounter::implRelease();
		}
	};

	// Same data as in the Whisper.sCaptureParams public structure, the durations are scaled from FP32 seconds into uint32_t samples at 16 kHz
	struct CaptureParams
	{
		uint32_t minDuration, maxDuration, dropStartSilence, pauseDuration;
		uint32_t flags;

		CaptureParams( const sCaptureParams& cp )
		{
			// Convert these floats from seconds to samples
			__m128 floats = _mm_loadu_ps( &cp.minDuration );
			floats = _mm_mul_ps( floats, _mm_set1_ps( (float)SAMPLE_RATE ) );
			floats = _mm_round_ps( floats, _MM_FROUND_NINT );
			__m128i ints = _mm_cvtps_epi32( floats );
			store16( &minDuration, ints );

			flags = cp.flags;
		}
	};

	class Capture
	{
		CComPtr<IMFSourceReader> reader;
		const CaptureParams captureParams;
		const sCaptureCallbacks callbacks;
		// Count of channels delivered from the source reader
		uint8_t readerChannels = 0;
		volatile char stateFlags = 0;

		PTP_WORK work = nullptr;
		volatile HRESULT workStatus = S_OK;

		TranscribeBufferObj buffer;
		CComAutoCriticalSection critSec;
		AudioBuffer pcm;
		AudioBuffer::pfnAppendSamples pfnAppendSamples = nullptr;
		int64_t pcmStartTime = 0;
		int64_t nextSampleTime = 0;
		VAD vad;
		sFullParams fullParams;
		ProfileCollection& profiler;
		iContext* const whisperContext;

		// Set the state bit, and if needed notify user with the callback.
		HRESULT setStateFlag( eCaptureStatus newBit ) noexcept
		{
			const uint8_t bit = (uint8_t)newBit;
			const uint8_t oldVal = (uint8_t)InterlockedOr8( &stateFlags, (char)bit );
			if( nullptr == callbacks.captureStatus )
				return S_OK;	// no callbacks
			if( 0 != ( oldVal & bit ) )
				return S_OK;	// The bit was already set
			return callbacks.captureStatus( callbacks.pv, (eCaptureStatus)( oldVal | bit ) );
		}

		// Clear the state bit, and if needed notify user with the callback
		HRESULT clearStateFlag( eCaptureStatus clearBit ) noexcept
		{
			const uint8_t bit = (uint8_t)clearBit;
			const uint8_t mask = ~bit;
			const uint8_t oldVal = (uint8_t)InterlockedAnd8( &stateFlags, (char)mask );
			if( nullptr == callbacks.captureStatus )
				return S_OK;	// no callbacks
			if( 0 == ( oldVal & bit ) )
				return S_OK;	// The bit wasn't there
			return callbacks.captureStatus( callbacks.pv, (eCaptureStatus)( oldVal & mask ) );
		}

		bool hasStateFlag( eCaptureStatus testBit ) const
		{
			const uint8_t bit = (uint8_t)testBit;
			return 0 != ( (uint8_t)stateFlags & bit );
		}

		HRESULT workCallback();
		static void __stdcall callbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID pv, PTP_WORK Work );

		HRESULT readSample( bool discard );

		// Run voice detection on the data in pcm.mono vector.
		// When not detected, return 0. When detected, return last frame index where it is detected.
		size_t detectVoice();

		HRESULT postPoolWork()
		{
			assert( workStatus == S_OK );
			CHECK( setStateFlag( eCaptureStatus::Transcribing ) );

			workStatus = S_FALSE;
			buffer.currentOffset = pcmStartTime;
			pcm.swap( buffer.pcm );
			SubmitThreadpoolWork( work );
			pcmStartTime = nextSampleTime;
			pcm.clear();
			vad.clear();
			return S_OK;
		}

	public:
		Capture( const sCaptureCallbacks& cb, const iAudioCapture* ac, const sFullParams& sfp, iContext* wc, ProfileCollection& pc ) :
			callbacks( cb ),
			captureParams( ac->getParams() ),
			fullParams( sfp ), whisperContext( wc ), profiler( pc )
		{
		}

		~Capture()
		{
			if( workStatus == S_FALSE && nullptr != work )
				WaitForThreadpoolWorkCallbacks( work, FALSE );

			if( nullptr != work )
			{
				CloseThreadpoolWork( work );
				work = nullptr;
			}
		}

		HRESULT startup( const iAudioCapture* ac );

		HRESULT checkCancel() noexcept
		{
			if( nullptr == callbacks.shouldCancel )
				return S_OK;
			return callbacks.shouldCancel( callbacks.pv );
		}

		HRESULT run();
	};

	HRESULT Capture::startup( const iAudioCapture* ac )
	{
		// Initialize the MF source reader
		CHECK( ac->getReader( &reader ) );
		work = CreateThreadpoolWork( &callbackStatic, this, nullptr );
		if( nullptr == work )
			return HRESULT_FROM_WIN32( GetLastError() );

		// Set up media type, and figure out sample handler
		CHECK( reader->SetStreamSelection( MF_SOURCE_READER_ALL_STREAMS, FALSE ) );
		CHECK( reader->SetStreamSelection( MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE ) );

		CComPtr<IMFMediaType> mtNative;
		CHECK( reader->GetNativeMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, MF_SOURCE_READER_CURRENT_TYPE_INDEX, &mtNative ) );
		UINT32 numChannels;
		CHECK( mtNative->GetUINT32( MF_MT_AUDIO_NUM_CHANNELS, &numChannels ) );

		const bool sourceMono = numChannels < 2;
		const bool wantStereo = 0 != ( captureParams.flags & (uint32_t)eCaptureFlags::Stereo );
		pfnAppendSamples = AudioBuffer::appendSamplesFunc( sourceMono, wantStereo );

		CComPtr<IMFMediaType> mt;
		this->readerChannels = ( !sourceMono && wantStereo ) ? 2 : 1;
		CHECK( createMediaType( !sourceMono, &mt ) );
		CHECK( reader->SetCurrentMediaType( MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, mt ) );

		CHECK( setStateFlag( eCaptureStatus::Listening ) );
		return S_OK;
	}

	// This method is called in a loop until user stops the audio capture
	HRESULT Capture::run()
	{
		HRESULT hr;
		if( hasStateFlag( eCaptureStatus::Stalled ) )
		{
			hr = workStatus;
			CHECK( hr );
			if( S_OK != hr )
			{
				// Still stalled, discard the upcoming sample
				return readSample( true );
			}
			else
			{
				// The postponed task has completed by now, no longer stalled
				// Move the current PCM buffer to the transcribe thread
				CHECK( clearStateFlag( eCaptureStatus::Stalled ) );
				return postPoolWork();
			}
		}

		const size_t oldSamples = pcm.mono.size();
		CHECK( readSample( false ) );
		const size_t newSamples = pcm.mono.size();

		const size_t lastVoiceFrame = detectVoice();
		if( lastVoiceFrame == 0 )
		{
			// No voice is detected in the entire buffered audio
			clearStateFlag( eCaptureStatus::Voice );
			if( newSamples < captureParams.dropStartSilence )
				return S_OK;

			pcm.clear();
			vad.clear();
			pcmStartTime = nextSampleTime;
			return S_OK;
		}

		const bool newFrameVoice = lastVoiceFrame + captureParams.pauseDuration >= oldSamples;

		if( newFrameVoice )
		{
			// A voice is detected in the buffer, and it was fairly recently
			setStateFlag( eCaptureStatus::Voice );
			if( newSamples < captureParams.maxDuration )
				return S_OK;	// While voice is continuously detected, we allow to grow the buffer up to `maxDuration` time
		}
		else
		{
			// A voice is detected in the buffer, but it was a while ago
			clearStateFlag( eCaptureStatus::Voice );
			if( newSamples < captureParams.minDuration )
				return S_OK;	// When detected pause in the voice, we fire the transcribe task right away.
		}

		// Hopefully, we have enough captured PCM data to run the ASR model.
		// Check the background task status first.
		hr = workStatus;
		CHECK( hr );
		if( hr == S_OK )
		{
			// S_OK workStatus means the previously posted transcribe job has completed successfully by now
			return postPoolWork();
		}

		// S_FALSE means the previously posted transcribe job is still running
		// Allow the buffer to grow up to maxDuration length, before starting to drop the samples
		if( newSamples < captureParams.maxDuration )
			return S_OK;

		// The previous task has not finished yet, but we don't want to grow the buffer even further.
		// We don't want concurrent transcribes here because not implemented, will simply crash.
		// Set the "Stalled" flag which causes capture to drop further samples
		setStateFlag( eCaptureStatus::Stalled );
		return S_OK;
	}

	HRESULT Capture::readSample( bool discard )
	{
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
				logError( u8"Media type changes ain’t supported by the library." );
				return E_UNEXPECTED;
			}

			if( dwFlags & MF_SOURCE_READERF_ENDOFSTREAM )
				return E_EOF;

			if( !sample )
				continue;

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
				if( !discard )
				{
					const size_t prevSize = pcm.mono.size();
					( pcm.*pfnAppendSamples )( pAudioData, countFloats );
					const size_t newSize = pcm.mono.size();
					this->nextSampleTime += ( newSize - prevSize );
				}
				else
				{
					this->nextSampleTime += countFloats / readerChannels;
				}
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

	HRESULT Capture::workCallback()
	{
		CHECK( whisperContext->runFull( fullParams, &buffer ) );
		CHECK( clearStateFlag( eCaptureStatus::Transcribing ) );
		return S_OK;
	}

	void __stdcall Capture::callbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID pv, PTP_WORK Work )
	{
		Capture* pThis = (Capture*)pv;
		HRESULT status = E_UNEXPECTED;
		try
		{
			status = pThis->workCallback();
		}
		catch( HRESULT hr )
		{
			status = hr;
		}
		catch( const std::bad_alloc& )
		{
			status = E_OUTOFMEMORY;
		}
		catch( const std::exception& )
		{
			status = E_FAIL;
		}
		assert( S_OK == status || FAILED( status ) );
		pThis->workStatus = status;
	}

	size_t Capture::detectVoice()
	{
		auto pf = profiler.cpuBlock( eCpuBlock::VAD );
		return vad.detect( pcm.mono.data(), pcm.mono.size() );
	}
}

HRESULT COMLIGHTCALL ContextImpl::runCapture( const sFullParams& params, const sCaptureCallbacks& callbacks, const iAudioCapture* reader )
{
	if( nullptr == reader )
		return E_POINTER;

	// Validate a few things
	{
		const auto& cp = reader->getParams();
		if( cp.minDuration < 0.125f || cp.minDuration > 30.0f )
		{
			logError( u8"%s parameter %g is out of range", "minDuration", cp.minDuration );
			return E_INVALIDARG;
		}
		if( cp.maxDuration < 0.125f || cp.maxDuration > 30.0f )
		{
			logError( u8"%s parameter %g is out of range", "maxDuration", cp.maxDuration );
			return E_INVALIDARG;
		}
	}

	auto profCompleteCpu = profiler.cpuBlock( eCpuBlock::RunComplete );
	Capture capture{ callbacks, reader, params, this, profiler };
	CHECK( capture.startup( reader ) );

	while( true )
	{
		HRESULT hr = capture.checkCancel();
		CHECK( hr );
		if( hr != S_OK )
			return S_OK;
		CHECK( capture.run() );
	}
}