#include "stdafx.h"
#include "ContextImpl.h"
using namespace Whisper;

namespace
{
	// Offset the timestamp with mediaTimeOffset to find the time relative to the start of the iSpectrogram buffer,
	// then scale from 100 nanosecond ticks into the Whisper's 10ms chunks, rounding down
	inline int64_t chunkOffset( int64_t time, int64_t mediaTimeOffset )
	{
		time -= mediaTimeOffset;
		return ( time * 100 ) / 10'000'000;
	}

	// Compute per-channel sum of std::absf( pcm ) in the specified buffer,
	// and return left / right numbers in the lower 2 lanes of the SSE vector
	inline __m128 __vectorcall computeChannelsEnergy( const std::vector<StereoSample>& sourceVector )
	{
		// Might be possible to implement way more sophisticated, and precise, version of this function.
		// For example, compute these 3 metrics with VAD code, and cluster the numbers somehow.
		// Not doing that currently; instead, replicating the simple version from the whisper.cpp original version.

		const StereoSample* rsi = sourceVector.data();
		const StereoSample* const rsiEnd = rsi + sourceVector.size();
		const StereoSample* const rsiEndAligned = rsi + ( sourceVector.size() & ( ~(size_t)1 ) );

		// Move 0x7FFFFFFF to lowest lane of the int32 vector;
		// unlike float scalars or all vectors, integer scalar constants are in the instruction stream
		__m128i absMaskInt = _mm_cvtsi32_si128( (int)0x7FFFFFFFu );
		// Broadcast over the complete vector
		absMaskInt = _mm_shuffle_epi32( absMaskInt, 0 );
		// Bitcast to FP32 vector, for _mm_and_ps instruction
		const __m128 absMask = _mm_castsi128_ps( absMaskInt );

		__m128 acc = _mm_setzero_ps();
		for( ; rsi < rsiEndAligned; rsi += 2 )
		{
			__m128 v = _mm_loadu_ps( (const float*)rsi );
			v = _mm_and_ps( v, absMask );
			acc = _mm_add_ps( acc, v );
		}
		if( rsi != rsiEnd )
		{
			__m128 v = _mm_castpd_ps( _mm_load_sd( (const double*)rsi ) );
			v = _mm_and_ps( v, absMask );
			acc = _mm_add_ps( acc, v );
		}

		// Return acc.xy + acc.zw
		acc = _mm_add_ps( acc, _mm_movehl_ps( acc, acc ) );
		return acc;
	}

	inline eSpeakerChannel produceResult( const __m128 ev )
	{
		// Original code did following:
		// if( energy0 > 1.1 * energy1 ) speaker = "(speaker 0)"; else if( energy1 > 1.1 * energy0 ) speaker = "(speaker 1)"; else speaker = "(speaker ?)";

		// Flip left/right channels
		__m128 tmp = _mm_shuffle_ps( ev, ev, _MM_SHUFFLE( 3, 2, 0, 1 ) );
		// Multiply by the magic number
		tmp = _mm_mul_ps( tmp, _mm_set1_ps( 1.1f ) );
		// Compare for ev > tmp
		tmp = _mm_cmpgt_ps( ev, tmp );
		const uint32_t mask = (uint32_t)_mm_movemask_ps( tmp ) & 0b11;

		assert( mask != 0b11 );	// That would mean the following is true: ( ( left > right * 1.1 ) && ( right > left * 1.1 ) )

		return (eSpeakerChannel)mask;
	}
}

HRESULT COMLIGHTCALL ContextImpl::detectSpeaker( const sTimeInterval& time, eSpeakerChannel& result ) const noexcept
{
	// Ensure we have the spectrogram
	if( nullptr == currentSpectrogram )
	{
		logError( u8"Because the audio is streamed, iContext.detectSpeaker() method only works when called from the callbacks" );
		return OLE_E_BLANK;
	}

	// Load the timestamps
	int64_t begin = (int64_t)time.begin.ticks;
	int64_t end = (int64_t)time.end.ticks;
	// Offset + scale into chunks
	begin = chunkOffset( begin, mediaTimeOffset );
	end = chunkOffset( end, mediaTimeOffset );

	int64_t len = end - begin;
	if( len <= 0 )
	{
		result = eSpeakerChannel::Unsure;
		return S_OK;
	}

	// Extract the slice of stereo PCM data
	HRESULT hr = currentSpectrogram->copyStereoPcm( (size_t)begin, (size_t)len, diarizeBuffer );
	if( hr == OLE_E_BLANK )
	{
		result = eSpeakerChannel::NoStereoData;
		return S_OK;
	}
	CHECK( hr );

	const __m128 energyVec = computeChannelsEnergy( diarizeBuffer );
	result = produceResult( energyVec );
	return S_OK;
}