#include "stdafx.h"
#include "AudioBuffer.h"
using namespace Whisper;

void AudioBuffer::appendMono( const float* rsi, size_t countFloats )
{
	mono.insert( mono.end(), rsi, rsi + countFloats );
}

void AudioBuffer::appendStereo( const float* rsi, size_t countFloats )
{
	assert( 0 == ( countFloats % 2 ) );
	const size_t countSamples = countFloats / 2;

	const size_t oldLength = mono.size();
	assert( oldLength * 2 == stereo.size() );
	mono.resize( oldLength + countSamples );
	stereo.resize( ( oldLength + countSamples ) * 2 );

	const float* const rsiEnd = rsi + countSamples * 2;
	const float* const rsiEndAligned = rsiEnd - ( countSamples * 2 ) % 8;

	float* rdiStereo = &stereo[ oldLength * 2 ];
	float* rdiMono = &mono[ oldLength ];

	const __m128 half = _mm_set1_ps( 0.5f );
	for( ; rsi < rsiEndAligned; rsi += 8, rdiStereo += 8, rdiMono += 4 )
	{
		// Load 4 samples = 8 floats 
		__m128 v0 = _mm_loadu_ps( rsi );	// L0, R0, L1, R1
		__m128 v1 = _mm_loadu_ps( rsi + 4 );// L2, R2, L3, R3

		// Store into the stereo PCM vector
		_mm_storeu_ps( rdiStereo, v0 );
		_mm_storeu_ps( rdiStereo + 4, v1 );

		// Compute and store the average of these channels
		__m128 left = _mm_shuffle_ps( v0, v1, _MM_SHUFFLE( 2, 0, 2, 0 ) );
		__m128 right = _mm_shuffle_ps( v0, v1, _MM_SHUFFLE( 3, 1, 3, 1 ) );
		__m128 sum = _mm_add_ps( left, right );
		sum = _mm_mul_ps( sum, half );
		_mm_storeu_ps( rdiMono, sum );
	}

#pragma loop (no_vector)
	for( ; rsi < rsiEnd; rsi += 2, rdiStereo += 2, rdiMono++ )
	{
		__m128 vec = _mm_castpd_ps( _mm_load_sd( (const double*)rsi ) );
		_mm_store_sd( (double*)rdiStereo, _mm_castps_pd( vec ) );

		vec = _mm_add_ss( vec, _mm_movehdup_ps( vec ) );
		vec = _mm_mul_ss( vec, half );
		_mm_store_ss( rdiMono, vec );
	}
}

void AudioBuffer::appendDownmixedStereo( const float* rsi, size_t countFloats )
{
	assert( 0 == ( countFloats % 2 ) );
	const size_t countSamples = countFloats / 2;

	const size_t oldLength = mono.size();
	mono.resize( oldLength + countSamples );

	const float* const rsiEnd = rsi + countSamples * 2;
	const float* const rsiEndAligned = rsiEnd - ( countSamples * 2 ) % 8;

	float* rdiMono = &mono[ oldLength ];

	const __m128 half = _mm_set1_ps( 0.5f );
	for( ; rsi < rsiEndAligned; rsi += 8, rdiMono += 4 )
	{
		// Load 4 samples = 8 floats 
		__m128 v0 = _mm_loadu_ps( rsi );	// L0, R0, L1, R1
		__m128 v1 = _mm_loadu_ps( rsi + 4 );// L2, R2, L3, R3

		// Compute and store the average of these channels
		__m128 left = _mm_shuffle_ps( v0, v1, _MM_SHUFFLE( 2, 0, 2, 0 ) );
		__m128 right = _mm_shuffle_ps( v0, v1, _MM_SHUFFLE( 3, 1, 3, 1 ) );
		__m128 sum = _mm_add_ps( left, right );
		sum = _mm_mul_ps( sum, half );
		_mm_storeu_ps( rdiMono, sum );
	}

#pragma loop (no_vector)
	for( ; rsi < rsiEnd; rsi += 2, rdiMono++ )
	{
		__m128 vec = _mm_castpd_ps( _mm_load_sd( (const double*)rsi ) );
		vec = _mm_add_ss( vec, _mm_movehdup_ps( vec ) );
		vec = _mm_mul_ss( vec, half );
		_mm_store_ss( rdiMono, vec );
	}
}