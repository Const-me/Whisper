#pragma once
#include "WhisperModel.h"
#include "iSpectrogram.h"
#include "audioConstants.h"

namespace Whisper
{
	struct iAudioBuffer;

	// This implementation of iSpectrogram interface converts complete audio into MEL spectrogram
	// Used for unbuffered audio, and capture: iContext.runFull and runCapture methods.
	class Spectrogram: public iSpectrogram
	{
		uint32_t length = 0;
		static constexpr uint32_t mel = N_MEL;
		std::vector<float> data;
		std::vector<StereoSample> stereo;

		HRESULT makeBuffer( size_t off, size_t len, const float** buffer, size_t& stride ) noexcept override final
		{
			if( off + len > length )
				return E_BOUNDS;
			*buffer = &data[ off ];
			stride = length;
			return S_OK;
		}

		class MelContext;

		HRESULT copyStereoPcm( size_t offset, size_t length, std::vector<StereoSample>& buffer ) const override final;

	public:
		size_t getLength() const noexcept override final
		{
			return length;
		}
		HRESULT pcmToMel( const iAudioBuffer* buffer, const Filters& filters, int threads = 1 );

		size_t memoryUsage() const
		{
			return data.size() * 4;
		}
	};

	// average the fabs of the signal
	void computeSignalEnergy( std::vector<float>& result, const iAudioBuffer* buffer, int n_samples_per_half_window );
}