#pragma once
#include <complex>
#include <memory>
#include "audioConstants.h"

namespace Whisper
{
	class VAD
	{
		using cplx = std::complex<float>;
		std::unique_ptr<cplx[]> fft_signal;

		struct Feature
		{
			float energy;
			float F;
			float SFM;
		};
		const Feature primThresh;
		static Feature defaultPrimaryThresholds();

		struct State
		{
			Feature currThresh;
			Feature minFeature;
			Feature curr;

			uint32_t lastSpeech;
			float silenceRun;
			uint32_t i;
		};
		State state;

		static inline void fft( cplx* buf, cplx* out, size_t n, size_t step );
		void fft() const;

		static float computeEnergy( const float* rsi );
		static float computeDominant( const cplx* spectrum );
		static float computreSpectralFlatnessMeasure( const cplx* spectrum );

	public:

		VAD();

		// When no speech is detected, returns 0
		// When speech is detected, returns sample position for the end of the speech
		size_t detect( const float* rsi, size_t length );

		void clear();

		static constexpr uint32_t FFT_POINTS = 256;
		static constexpr float FFT_STEP = (float)SAMPLE_RATE / (float)FFT_POINTS;
	};
}