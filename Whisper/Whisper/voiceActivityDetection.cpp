#include "stdafx.h"
#include "voiceActivityDetection.h"
using namespace Whisper;

// Initially ported (poorly) from there https://github.com/panmasuo/voice-activity-detection MIT license
// The code is based on that article:
// https://www.researchgate.net/publication/255667085_A_simple_but_efficient_real-time_voice_activity_detection_algorithm

inline VAD::Feature VAD::defaultPrimaryThresholds()
{
	Feature f;
	f.energy = 40;
	f.F = 185;
	f.SFM = 5;
	return f;
}

VAD::VAD() :
	primThresh( defaultPrimaryThresholds() )
{
	fft_signal = std::make_unique<cplx[]>( FFT_POINTS );
}

inline void VAD::fft( cplx* buf, cplx* out, size_t n, size_t step )
{
	if( step < n )
	{
		fft( out, buf, n, step * 2 );
		fft( out + step, buf + step, n, step * 2 );

		for( size_t i = 0; i < n; i += 2 * step )
		{
			// cplx t = cexp(-I * M_PI * i / n) * out[i + step];
			// using namespace std::complex_literals;
			const float mul = (float)M_PI * (float)(int)i / (float)(int)n;
			// cplx t0 = std::exp( -1.0if * mul ) * out[ i + step ];
			float sine, cosine;
			DirectX::XMScalarSinCos( &sine, &cosine, -mul );
			const cplx exponent{ cosine, sine };
			cplx t = exponent * out[ i + step ];

			buf[ i / 2 ] = out[ i ] + t;
			buf[ ( i + n ) / 2 ] = out[ i ] - t;
		}
	}
}

void VAD::fft() const
{
	cplx out[ FFT_POINTS ];
	memcpy( &out[ 0 ], fft_signal.get(), FFT_POINTS * sizeof( cplx ) );

	fft( fft_signal.get(), out, FFT_POINTS, 1 );
}

constexpr float mulInt16FromFloat = 32768.0;

inline float squareRoot( float x )
{
	__m128 v = _mm_set_ss( x );
	v = _mm_sqrt_ss( v );
	return _mm_cvtss_f32( v );
}

float VAD::computeEnergy( const float* rsi )
{
	// calculate_energy
	double sum = 0;
	for( size_t i = 0; i < FFT_POINTS; i++ )
	{
		float f = rsi[ i ];
		f *= mulInt16FromFloat;
		f *= f;
		sum += f;
	}
	return squareRoot( (float)( sum * ( 1.0 / FFT_POINTS ) ) );
}

float VAD::computeDominant( const cplx* spectrum )
{
	// calculate_dominant, reworked heavily
	float maxMagSquared = 0;
	int maxFreq = 0;

	for( int i = 0; i < FFT_POINTS / 2; i++ )
	{
		const float real = (float)spectrum[ i ].real();
		const float imag = (float)spectrum[ i ].imag();
		float sq = real * real + imag * imag;
		if( sq <= maxMagSquared )
			continue;
		maxMagSquared = sq;
		maxFreq = i;
	}
	return (float)maxFreq * FFT_STEP;
}

float VAD::computreSpectralFlatnessMeasure( const cplx* spectrum )
{
	// calculate_sfm
	double sum_ari = 0;
	double sum_geo = 0;
	for( size_t i = 0; i < FFT_POINTS; i++ )
	{
		// sig = cabsf( spectrum[ i ] );
		float sig = std::abs( spectrum[ i ] );
		sum_ari += sig;
		sum_geo += std::log( sig );
	}
	sum_ari = sum_ari / FFT_POINTS;
	sum_geo = std::exp( sum_geo / FFT_POINTS );
	return -10.0f * std::log10f( (float)( sum_geo / sum_ari ) );
}

void VAD::clear()
{
	memset( &state, 0, sizeof( State ) );
	state.currThresh = primThresh;
}

size_t VAD::detect( const float* rsi, size_t length )
{
	// The cryptic numbers in the comments are from section 3 "Proposed VAD Algorithm" of the article, on page 2550, on the right
	const size_t frames = length / FFT_POINTS;
	if( frames <= 0 )
	{
		clear();
		return 0;
	}

	// Load detection state from the field
	Feature currThresh = state.currThresh;
	Feature minFeature = state.minFeature;
	Feature curr = state.curr;

	size_t lastSpeech = state.lastSpeech;
	float silenceRun = state.silenceRun;
	size_t i = state.i;

	// Run the loop just on the [ state.i .. frames ] slice of the input PCM
	rsi += i * FFT_POINTS;
	for( ; i < frames; i++, rsi += FFT_POINTS )
	{
		// 3-2 calculate FFT
		for( size_t j = 0; j < FFT_POINTS; j++ )
		{
			const float re = rsi[ j ] * mulInt16FromFloat;
			fft_signal[ j ] = { re, 0.0f };
		}
		fft();

		// 3-1 + 3-2 calculate features
		curr.energy = computeEnergy( rsi );
		curr.F = computeDominant( fft_signal.get() );
		curr.SFM = computreSpectralFlatnessMeasure( fft_signal.get() );

		// 3-3 calculate minimum value for first 30 frames
		if( i == 0 )
			minFeature = curr;
		else if( i < 30 )
		{
			minFeature.energy = std::min( minFeature.energy, curr.energy );
			minFeature.F = std::min( minFeature.F, curr.F );
			minFeature.SFM = std::min( minFeature.SFM, curr.SFM );
		}

		// 3-4 set thresholds
		currThresh.energy = primThresh.energy * std::log10f( minFeature.energy );

		// 3-5 calculate decision
		uint8_t counter = 0;
		if( ( curr.energy - minFeature.energy ) >= currThresh.energy )
			counter = 1;
		if( ( curr.F - minFeature.F ) >= currThresh.F )
			counter++;
		if( ( curr.SFM - minFeature.SFM ) >= currThresh.SFM )
			counter++;

		if( counter > 1 )
		{
			// 3-6 If counter > 1 mark the current frame as speech
			lastSpeech = ( i + 1 ) * FFT_POINTS;
			silenceRun = 0.0f;
		}
		else
		{
			silenceRun += 1.0f;
			// 3-7 If current frame is marked as silence, update the energy minimum value
			minFeature.energy = ( ( silenceRun * minFeature.energy ) + curr.energy ) / ( silenceRun + 1 );
		}

		// 3-8
		currThresh.energy = primThresh.energy * std::log10f( minFeature.energy );
	}

	// Store the updated detection state back into that field
	state.currThresh = currThresh;
	state.minFeature = minFeature;
	state.curr = curr;
	state.lastSpeech = (uint32_t)lastSpeech;
	state.silenceRun = silenceRun;
	state.i = (uint32_t)i;

	return lastSpeech;
}