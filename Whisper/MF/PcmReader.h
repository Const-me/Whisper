#pragma once
#include "../Whisper/audioConstants.h"
#include <mfidl.h>
#include <mfreadwrite.h>
#include "AudioBuffer.h"
#include "../API/iMediaFoundation.cl.h"

namespace Whisper
{
	// PCM buffer with 10 milliseconds of single-channel audio
	struct PcmMonoChunk
	{
		std::array<float, FFT_STEP> mono;
	};
	// PCM buffer with 10 milliseconds of interleaved stereo
	struct PcmStereoChunk
	{
		std::array<float, FFT_STEP * 2> stereo;
	};

	__interface iSampleHandler;

	constexpr HRESULT E_EOF = HRESULT_FROM_WIN32( ERROR_HANDLE_EOF );

	// Utility class which reads chunks of FFT_STEP FP32 PCM samples from the MF source reader
	// The class always delivers mono chunks, and can optionally deliver stereo in a separate buffer.
	class PcmReader
	{
		// A small intermediate buffer with PCM data for complete media foundation samples
		AudioBuffer pcm;
		// Index of the first unconsumed sample in the pcm buffer
		size_t bufferReadOffset = 0;
		// Utility object to abstract away mono versus stereo shenanigans
		const iSampleHandler* sampleHandler;
		// The underlying MF source reader which delivers audio data
		CComPtr<IMFSourceReader> reader;
		// True after we consumed all available media samples from the reader
		bool m_readerEndOfFile = false;
		// True if this object delivers stereo samples
		bool m_stereoOutput = false;
		// The count of chunks we expect to get from the reader
		size_t m_length = 0;
		// Read next sample from the reader, store in the PCM buffer in this class
		HRESULT readNextSample();

	public:

		PcmReader( const iAudioReader* reader );

		// Count of chunks in the MEL spectrogram.
		// The PCM audio is generally slightly longer than that, due to the incomplete last chunk.
		size_t getLength() const noexcept
		{
			return m_length;
		}

		// True when the stereo flag passed to constructor, and the audio stream actually has 2 or more audio channels
		bool outputsStereo() const { return m_stereoOutput; }

		// Load another 10ms chunk from the stream
		// For the last chunk in the stream, the output buffers are padded with zeros
		HRESULT readChunk( PcmMonoChunk& mono, PcmStereoChunk* stereo );
	};
}