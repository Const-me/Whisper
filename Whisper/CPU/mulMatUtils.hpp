#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <assert.h>

__forceinline __m128i f16Load( const uint16_t* rsi )
{
	return _mm_loadu_si128( ( const __m128i* )rsi );
}

constexpr size_t maskAlign8 = ~(size_t)7;

__forceinline void transpose8( uint16_t* rdi, size_t w, const uint16_t* rsi, size_t sourceStride, size_t destStride )
{
	assert( 0 == ( (size_t)rdi ) % 16 );
	assert( 0 == destStride % 8 );
	assert( w <= sourceStride );

	const uint16_t* const rsiEndAligned = rsi + ( w & maskAlign8 );
	const uint16_t* rsi5 = rsi + sourceStride * 5;
	uint16_t* rdi5 = rdi + destStride * 5;
	const size_t rem = w % 8;
	for( ; rsi < rsiEndAligned; rsi += 8, rsi5 += 8, rdi += 8 * destStride, rdi5 += 8 * destStride )
	{
		// Load 8x8 block into 8 registers
		__m128i r0 = f16Load( rsi );                     // 00, 01, 02, 03, 04, 05, 06, 07
		__m128i r1 = f16Load( rsi + sourceStride );      // 10, 11, 12, 13, 14, 15, 16, 17
		__m128i r2 = f16Load( rsi + sourceStride * 2 );  // 20, 21, 22, 23, 24, 25, 26, 27
		__m128i r3 = f16Load( rsi5 - sourceStride * 2 ); // 30, 31, 32, 33, 34, 35, 36, 37
		__m128i r4 = f16Load( rsi5 - sourceStride );     // 40, 41, 42, 43, 44, 45, 46, 47
		__m128i r5 = f16Load( rsi5 );                    // 50, 51, 52, 53, 54, 55, 56, 57
		__m128i r6 = f16Load( rsi5 + sourceStride );     // 60, 61, 62, 63, 64, 65, 66, 67
		__m128i r7 = f16Load( rsi5 + sourceStride * 2 ); // 70, 71, 72, 73, 74, 75, 76, 77

		// Transpose FP16 values in registers
		__m128i t0 = _mm_unpacklo_epi16( r0, r1 ); // 00, 10, 01, 11, 02, 12, 03, 13
		__m128i t1 = _mm_unpackhi_epi16( r0, r1 ); // 04, 14, 05, 15, 06, 16, 07, 17
		__m128i t2 = _mm_unpacklo_epi16( r2, r3 ); // 20, 30, 21, 31, 22, 32, 23, 33
		__m128i t3 = _mm_unpackhi_epi16( r2, r3 ); // 24, 34, 25, 35, 26, 36, 27, 37
		__m128i t4 = _mm_unpacklo_epi16( r4, r5 ); // 40, 50, 41, 52, 42, 52, 43, 53
		__m128i t5 = _mm_unpackhi_epi16( r4, r5 ); // 44, 54, 45, 55, 46, 56, 47, 57
		__m128i t6 = _mm_unpacklo_epi16( r6, r7 ); // 60, 70, 61, 71, 62, 72, 63, 73
		__m128i t7 = _mm_unpackhi_epi16( r6, r7 ); // 64, 74, 65, 75, 66, 76, 67, 77

		r0 = _mm_unpacklo_epi32( t0, t2 ); // 00, 10, 20, 30, 01, 11, 21, 31
		r1 = _mm_unpackhi_epi32( t0, t2 ); // 02, 12, 22, 32, 03, 13, 23, 33
		r2 = _mm_unpacklo_epi32( t1, t3 ); // 04, 14, 24, 34, 05, 15, 25, 35
		r3 = _mm_unpackhi_epi32( t1, t3 ); // 06, 16, 26, 36, 07, 17, 27, 37
		r4 = _mm_unpacklo_epi32( t4, t6 ); // 40, 50, 60, 70, 41, 51, 61, 71
		r5 = _mm_unpackhi_epi32( t4, t6 ); // 42, 52, 62, 72, 43, 53, 63, 73
		r6 = _mm_unpacklo_epi32( t5, t7 ); // 44, 54, 64, 74, 45, 55, 65, 75
		r7 = _mm_unpackhi_epi32( t5, t7 ); // 46, 56, 66, 76, 47, 57, 67, 77

		t0 = _mm_unpacklo_epi64( r0, r4 ); // 00, 10, 20, 30, 40, 50, 60, 70
		t1 = _mm_unpackhi_epi64( r0, r4 ); // 01, 11, 21, 31, 41, 52, 61, 71
		t2 = _mm_unpacklo_epi64( r1, r5 ); // 02, 12, 22, 32, 42, 52, 62, 72
		t3 = _mm_unpackhi_epi64( r1, r5 ); // 03, 13, 23, 33, 43, 53, 63, 73
		t4 = _mm_unpacklo_epi64( r2, r6 );
		t5 = _mm_unpackhi_epi64( r2, r6 );
		t6 = _mm_unpacklo_epi64( r3, r7 );
		t7 = _mm_unpackhi_epi64( r3, r7 );

		// Store
		store16( rdi, t0 );
		store16( rdi + destStride, t1 );
		store16( rdi + destStride * 2, t2 );
		store16( rdi5 - destStride * 2, t3 );
		store16( rdi5 - destStride, t4 );
		store16( rdi5, t5 );
		store16( rdi5 + destStride, t6 );
		store16( rdi5 + destStride * 2, t7 );
	}

#pragma loop( no_vector )
	for( size_t i = 0; i < rem; rsi++, rsi5++, rdi += destStride )
	{
		const int16_t* p0 = (const int16_t*)rsi;
		const int16_t* p5 = (const int16_t*)rsi5;
		// Load a complete column into a vector
		__m128i v = _mm_cvtsi32_si128( *rsi );
		v = _mm_insert_epi16( v, *( p0 + sourceStride ), 1 );
		v = _mm_insert_epi16( v, *( p0 + sourceStride * 2 ), 2 );
		v = _mm_insert_epi16( v, *( p5 - sourceStride * 2 ), 3 );
		v = _mm_insert_epi16( v, *( p5 - sourceStride ), 4 );
		v = _mm_insert_epi16( v, *( p5 ), 5 );
		v = _mm_insert_epi16( v, *( p5 + sourceStride ), 6 );
		v = _mm_insert_epi16( v, *( p5 + sourceStride * 2 ), 7 );
		// Store 8 FP16 values
		store16( rdi, v );
	}
}

inline void transpose8Partial( uint16_t* rdi, size_t w, size_t h, const uint16_t* rsi, size_t sourceStride, size_t destStride )
{
	assert( 0 == ( (size_t)rdi ) % 16 );
	assert( 0 == destStride % 8 );
	assert( w <= sourceStride );
	assert( h > 0 && h < 8 );

	const uint16_t* const rsiEndAligned = rsi + ( w & maskAlign8 );
	const uint16_t* rsi5 = rsi + sourceStride * 5;
	uint16_t* rdi5 = rdi + destStride * 5;
	const size_t rem = w % 8;
	for( ; rsi < rsiEndAligned; rsi += 8, rsi5 += 8, rdi += 8 * destStride, rdi5 += 8 * destStride )
	{
		// Load the block into 8 registers, set unused rows to zero
		__m128i r0 = f16Load( rsi );
		__m128i r1 = _mm_setzero_si128();
		__m128i r2 = _mm_setzero_si128();
		__m128i r3 = _mm_setzero_si128();
		__m128i r4 = _mm_setzero_si128();
		__m128i r5 = _mm_setzero_si128();
		__m128i r6 = _mm_setzero_si128();
		// These branches, whether direct or indirect, are very predictable: same outcome for all iterations of the outer loop
		switch( h )
		{
		case 7:
			r6 = f16Load( rsi5 + sourceStride );
		case 6:
			r5 = f16Load( rsi5 );
		case 5:
			r4 = f16Load( rsi5 - sourceStride );
		case 4:
			r3 = f16Load( rsi5 - sourceStride * 2 );
		case 3:
			r2 = f16Load( rsi + sourceStride * 2 );
		case 2:
			r1 = f16Load( rsi + sourceStride );
		}
		__m128i r7 = _mm_setzero_si128();

		// Transpose FP16 values in registers
		__m128i t0 = _mm_unpacklo_epi16( r0, r1 ); // 00, 10, 01, 11, 02, 12, 03, 13
		__m128i t1 = _mm_unpackhi_epi16( r0, r1 ); // 04, 14, 05, 15, 06, 16, 07, 17
		__m128i t2 = _mm_unpacklo_epi16( r2, r3 ); // 20, 30, 21, 31, 22, 32, 23, 33
		__m128i t3 = _mm_unpackhi_epi16( r2, r3 ); // 24, 34, 25, 35, 26, 36, 27, 37
		__m128i t4 = _mm_unpacklo_epi16( r4, r5 ); // 40, 50, 41, 52, 42, 52, 43, 53
		__m128i t5 = _mm_unpackhi_epi16( r4, r5 ); // 44, 54, 45, 55, 46, 56, 47, 57
		__m128i t6 = _mm_unpacklo_epi16( r6, r7 ); // 60, 70, 61, 71, 62, 72, 63, 73
		__m128i t7 = _mm_unpackhi_epi16( r6, r7 ); // 64, 74, 65, 75, 66, 76, 67, 77

		r0 = _mm_unpacklo_epi32( t0, t2 ); // 00, 10, 20, 30, 01, 11, 21, 31
		r1 = _mm_unpackhi_epi32( t0, t2 ); // 02, 12, 22, 32, 03, 13, 23, 33
		r2 = _mm_unpacklo_epi32( t1, t3 ); // 04, 14, 24, 34, 05, 15, 25, 35
		r3 = _mm_unpackhi_epi32( t1, t3 ); // 06, 16, 26, 36, 07, 17, 27, 37
		r4 = _mm_unpacklo_epi32( t4, t6 ); // 40, 50, 60, 70, 41, 51, 61, 71
		r5 = _mm_unpackhi_epi32( t4, t6 ); // 42, 52, 62, 72, 43, 53, 63, 73
		r6 = _mm_unpacklo_epi32( t5, t7 ); // 44, 54, 64, 74, 45, 55, 65, 75
		r7 = _mm_unpackhi_epi32( t5, t7 ); // 46, 56, 66, 76, 47, 57, 67, 77

		t0 = _mm_unpacklo_epi64( r0, r4 ); // 00, 10, 20, 30, 40, 50, 60, 70
		t1 = _mm_unpackhi_epi64( r0, r4 ); // 01, 11, 21, 31, 41, 52, 61, 71
		t2 = _mm_unpacklo_epi64( r1, r5 ); // 02, 12, 22, 32, 42, 52, 62, 72
		t3 = _mm_unpackhi_epi64( r1, r5 ); // 03, 13, 23, 33, 43, 53, 63, 73
		t4 = _mm_unpacklo_epi64( r2, r6 );
		t5 = _mm_unpackhi_epi64( r2, r6 );
		t6 = _mm_unpacklo_epi64( r3, r7 );
		t7 = _mm_unpackhi_epi64( r3, r7 );

		// Store
		store16( rdi, t0 );
		store16( rdi + destStride, t1 );
		store16( rdi + destStride * 2, t2 );
		store16( rdi5 - destStride * 2, t3 );
		store16( rdi5 - destStride, t4 );
		store16( rdi5, t5 );
		store16( rdi5 + destStride, t6 );
		store16( rdi5 + destStride * 2, t7 );
	}

#pragma loop( no_vector )
	for( size_t i = 0; i < rem; rsi++, rsi5++, rdi += destStride )
	{
		const int16_t* p0 = (const int16_t*)rsi;
		const int16_t* p5 = (const int16_t*)rsi5;
		// Load a partial column into vector
		__m128i v = _mm_cvtsi32_si128( *rsi );
		switch( h )
		{
		case 7:
			v = _mm_insert_epi16( v, *( p5 + sourceStride ), 6 );
		case 6:
			v = _mm_insert_epi16( v, *( p5 ), 5 );
		case 5:
			v = _mm_insert_epi16( v, *( p5 - sourceStride ), 4 );
		case 4:
			v = _mm_insert_epi16( v, *( p5 - sourceStride * 2 ), 3 );
		case 3:
			v = _mm_insert_epi16( v, *( p0 + sourceStride * 2 ), 2 );
		case 2:
			v = _mm_insert_epi16( v, *( p0 + sourceStride ), 1 );
		}
		// Store 8 FP16 values
		store16( rdi, v );
	}
}

// Same as above, but skip the transpose. The source stride is distance between columns of the matrix.
__forceinline void copyColumnMajor( uint16_t* rdi, size_t w, const uint16_t* rsi, size_t sourceStride, size_t destStride )
{
	assert( 0 == ( (size_t)rdi ) % 16 );
	assert( 0 == destStride % 8 );

	constexpr size_t maskAlign4 = ~(size_t)3;

	const uint16_t* const rsiEndAligned = rsi + sourceStride * ( w & maskAlign4 );
	const uint16_t* const rsiEnd = rsi + sourceStride * w;
	for( ; rsi < rsiEndAligned; rsi += sourceStride * 4, rdi += destStride * 4 )
	{
		__m128i c = f16Load( rsi );
		store16( rdi, c );

		c = f16Load( rsi + sourceStride );
		store16( rdi + destStride, c );

		c = f16Load( rsi + sourceStride * 2 );
		store16( rdi + destStride * 2, c );

		c = f16Load( rsi + sourceStride * 3 );
		store16( rdi + destStride * 3, c );
	}

	for( ; rsi < rsiEnd; rsi += sourceStride, rdi += destStride )
	{
		__m128i c = f16Load( rsi );
		store16( rdi, c );
	}
}

__forceinline __m128i loadPartial( const uint16_t* x, size_t count )
{
	assert( count < 8 );
	__m128i ix;
	switch( count )
	{
	case 1: // load 2 bytes
		ix = _mm_cvtsi32_si128( *x );
		break;
	case 2: // load 4 bytes
		ix = _mm_cvtsi32_si128( *(const int*)x );
		break;
	case 3: // load 6 bytes
		ix = _mm_cvtsi32_si128( *(const int*)x );
		ix = _mm_insert_epi16( ix, x[ 2 ], 2 );
		break;
	case 4: // load 8 bytes
		ix = _mm_cvtsi64_si128( *(const int64_t*)x );
		break;
	case 5: // load 10 bytes
		ix = _mm_cvtsi64_si128( *(const int64_t*)x );
		ix = _mm_insert_epi16( ix, x[ 4 ], 4 );
		break;
	case 6: // load 12 bytes
		ix = _mm_cvtsi64_si128( *(const int64_t*)x );
		ix = _mm_insert_epi32( ix, *(const int*)( x + 4 ), 2 );
		break;
	case 7: // load 14 bytes
		ix = _mm_cvtsi64_si128( *(const int64_t*)x );
		ix = _mm_insert_epi32( ix, *(const int*)( x + 4 ), 2 );
		ix = _mm_insert_epi16( ix, x[ 6 ], 6 );
		break;
	default:
		return _mm_setzero_si128();
	}
	return ix;
}

inline void copyColumnMajorPartial( uint16_t* rdi, size_t w, size_t h, const uint16_t* rsi, size_t sourceStride, size_t destStride )
{
	assert( 0 == ( (size_t)rdi ) % 32 );
	assert( 0 == destStride % 8 );
	assert( h > 0 && h < 8 );

	const uint16_t* const rsiEnd = rsi + sourceStride * w;
	for( ; rsi < rsiEnd; rsi += sourceStride, rdi += destStride )
	{
		// Can't use mask loads because loading 2-byte elements
		// Still, that switch() in loadPartial makes a very predictable branch, same outcome for all iterations of this loop.
		__m128i c = loadPartial( rsi, h );
		store16( rdi, c );
	}
}

// Store zeros into block of memory, with aligned AVX store instructions
__forceinline void zeroAlignedMemory( void* pv, size_t cb )
{
	assert( 0 == cb % 16 );
	assert( 0 == ( (size_t)pv % 32 ) );

	uint8_t* rdi = (uint8_t*)pv;
	constexpr size_t maskAlign32 = ~(size_t)31;
	uint8_t* const rdiEndAligned = rdi + ( cb & maskAlign32 );
	uint8_t* const rdiEnd = rdi + cb;

	const __m256 zero = _mm256_setzero_ps();
	for( ; rdi < rdiEndAligned; rdi += 32 )
		_mm256_store_ps( (float*)rdi, zero );

	if( rdi < rdiEnd )
		_mm_store_ps( (float*)rdi, _mm_setzero_ps() );
}