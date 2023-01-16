#pragma once
#include "audioConstants.h"
#include "WhisperModel.h"
#include <memory>

namespace Whisper
{
	class HanningWindow
	{
		std::array<float, FFT_SIZE> hann;
	public:
		HanningWindow();

		float operator[]( size_t i ) const
		{
			return hann[ i ];
		}
	};

	extern const HanningWindow s_hanning;

	class SpectrogramContext
	{
		const Filters& filters;
		static float* fftRecursion( float* temp, const float* const rsi, const size_t len );
		std::unique_ptr<float[]> tempBuffer;

	public:
		SpectrogramContext( const Filters& flt );

		// First step of the MEL algorithm, and recursively compute the FFT
		void fft( std::array<float, N_MEL>& rdi, const float* pcm, size_t length );
	};
}