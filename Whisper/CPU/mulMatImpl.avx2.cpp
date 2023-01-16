#include "stdafx.h"
#include "mulMatImpl.h"
#include <immintrin.h>
#include "mulMatUtils.hpp"
using namespace CpuCompute;

namespace
{
	constexpr size_t prefetchBytes = 96;
	constexpr int prefetchHint = _MM_HINT_T0;

	constexpr size_t maskAlign16 = ~(size_t)15;

	__forceinline __m256i load( const void* rsi )
	{
		return _mm256_loadu_si256( ( const __m256i* )rsi );
	}

#define TRANSPOSE_8X16()                           \
                                                   \
	__m256i t0 = _mm256_unpacklo_epi16( r0, r1 );  \
	__m256i t1 = _mm256_unpackhi_epi16( r0, r1 );  \
	__m256i t2 = _mm256_unpacklo_epi16( r2, r3 );  \
	__m256i t3 = _mm256_unpackhi_epi16( r2, r3 );  \
	__m256i t4 = _mm256_unpacklo_epi16( r4, r5 );  \
	__m256i t5 = _mm256_unpackhi_epi16( r4, r5 );  \
	__m256i t6 = _mm256_unpacklo_epi16( r6, r7 );  \
	__m256i t7 = _mm256_unpackhi_epi16( r6, r7 );  \
                                                   \
	r0 = _mm256_unpacklo_epi32( t0, t2 );          \
	r1 = _mm256_unpackhi_epi32( t0, t2 );          \
	r2 = _mm256_unpacklo_epi32( t1, t3 );          \
	r3 = _mm256_unpackhi_epi32( t1, t3 );          \
	r4 = _mm256_unpacklo_epi32( t4, t6 );          \
	r5 = _mm256_unpackhi_epi32( t4, t6 );          \
	r6 = _mm256_unpacklo_epi32( t5, t7 );          \
	r7 = _mm256_unpackhi_epi32( t5, t7 );          \
                                                   \
	t0 = _mm256_unpacklo_epi64( r0, r4 );          \
	t1 = _mm256_unpackhi_epi64( r0, r4 );          \
	t2 = _mm256_unpacklo_epi64( r1, r5 );          \
	t3 = _mm256_unpackhi_epi64( r1, r5 );          \
	t4 = _mm256_unpacklo_epi64( r2, r6 );          \
	t5 = _mm256_unpackhi_epi64( r2, r6 );          \
	t6 = _mm256_unpacklo_epi64( r3, r7 );          \
	t7 = _mm256_unpackhi_epi64( r3, r7 )

	__forceinline void storeLow( void* rdi, __m256i v )
	{
		__m128i i = _mm256_castsi256_si128( v );
		_mm_store_si128( ( __m128i* )rdi, i );
	}

#define STORE_8X16_LOW()                     \
	storeLow( rdi, t0 );                     \
	storeLow( rdi + destStride, t1 );        \
	storeLow( rdi + destStride * 2, t2 );    \
	rdi += destStride * 8;                   \
	storeLow( rdiMid, t3 );                  \
	storeLow( rdiMid + destStride, t4 );     \
	storeLow( rdiMid + destStride * 2, t5 ); \
	rdiMid += destStride * 8;                \
	storeLow( rdiLast, t6 );                 \
	storeLow( rdiLast + destStride, t7 );    \
	rdiLast += destStride * 8

	__forceinline void storeHigh( void* rdi, __m256i v )
	{
		__m128i i = _mm256_extracti128_si256( v, 1 );
		_mm_store_si128( ( __m128i* )rdi, i );
	}

#define STORE_8X16_HIGH()                     \
	storeHigh( rdi, t0 );                     \
	storeHigh( rdi + destStride, t1 );        \
	storeHigh( rdi + destStride * 2, t2 );    \
	rdi += destStride * 8;                    \
	storeHigh( rdiMid, t3 );                  \
	storeHigh( rdiMid + destStride, t4 );     \
	storeHigh( rdiMid + destStride * 2, t5 ); \
	rdiMid += destStride * 8;                 \
	storeHigh( rdiLast, t6 );                 \
	storeHigh( rdiLast + destStride, t7 );    \
	rdiLast += destStride * 8

	__forceinline void prefetch( const uint8_t* p )
	{
		_mm_prefetch( (const char*)p, prefetchHint );
	}

	__forceinline void transpose8Avx2( uint16_t* rdiWords, size_t w, const uint16_t* rsiWords, size_t sourceStride, size_t destStride )
	{
		assert( 0 == ( (size_t)rdiWords ) % 16 );
		assert( 0 == destStride % 8 );
		assert( w <= sourceStride );

		// Scale strides to bytes, and cast the pointers
		sourceStride *= 2;
		destStride *= 2;
		uint8_t* rdi = (uint8_t*)rdiWords;
		const uint8_t* rsi = (const uint8_t*)rsiWords;

		const uint8_t* const rsiEndAligned = rsi + ( w & maskAlign16 ) * 2;
		const uint8_t* const rsiEnd = rsi + w * 2;
		const uint8_t* rsiMid = rsi + sourceStride * 3;
		const uint8_t* rsiLast = rsi + sourceStride * 6;
		uint8_t* rdiMid = rdi + destStride * 3;
		uint8_t* rdiLast = rdi + destStride * 6;

		while( rsi < rsiEndAligned )
		{
			// Load 16x8 block into 8 registers
			__m256i r0 = load( rsi );
			__m256i r1 = load( rsi + sourceStride );
			__m256i r2 = load( rsi + sourceStride * 2 );
			rsi += 32;
			__m256i r3 = load( rsiMid );
			__m256i r4 = load( rsiMid + sourceStride );
			__m256i r5 = load( rsiMid + sourceStride * 2 );
			rsiMid += 32;
			__m256i r6 = load( rsiLast );
			__m256i r7 = load( rsiLast + sourceStride );
			rsiLast += 32;

			// Transpose FP16 values in registers
			TRANSPOSE_8X16();

			// Store
			STORE_8X16_LOW();
			STORE_8X16_HIGH();

			if constexpr( prefetchBytes > 0 )
			{
				if( rsi + prefetchBytes < rsiEnd )
				{
					prefetch( rsi + prefetchBytes );
					prefetch( rsi + sourceStride + prefetchBytes );
					prefetch( rsi + sourceStride * 2 + prefetchBytes );
					prefetch( rsiMid + prefetchBytes );
					prefetch( rsiMid + sourceStride + prefetchBytes );
					prefetch( rsiMid + sourceStride * 2 + prefetchBytes );
					prefetch( rsiLast + prefetchBytes );
					prefetch( rsiLast + sourceStride + prefetchBytes );
				}
			}
		}

		if( rsi < rsiEnd )
		{
			// Loading 8 elements into corresponding lanes of 8 vectors
			// This way there's no data dependencies between these load instructions
			// Out of order execution should hopefully do it's magic in the CPU, running all these loads in parallel.
			__m128i r0;
			__m128i r1 = _mm_setzero_si128();
			__m128i r2 = _mm_setzero_si128();
			__m128i r3 = _mm_setzero_si128();
			__m128i r4 = _mm_setzero_si128();
			__m128i r5 = _mm_setzero_si128();
			__m128i r6 = _mm_setzero_si128();
			__m128i r7 = _mm_setzero_si128();

			__m128i t0, t1, t2, t3, t4, t5, t6;

#pragma loop( no_vector )
			while( rsi < rsiEnd )
			{
				r0 = _mm_cvtsi32_si128( *(const uint16_t*)rsi );
				r1 = _mm_insert_epi16( r1, *(const int16_t*)( rsi + sourceStride ), 1 );
				r2 = _mm_insert_epi16( r2, *(const int16_t*)( rsi + sourceStride * 2 ), 2 );
				rsi += 2;
				r3 = _mm_insert_epi16( r3, *(const int16_t*)( rsiMid ), 3 );
				r4 = _mm_insert_epi16( r4, *(const int16_t*)( rsiMid + sourceStride ), 4 );
				r5 = _mm_insert_epi16( r5, *(const int16_t*)( rsiMid + sourceStride * 2 ), 5 );
				rsiMid += 2;
				r6 = _mm_insert_epi16( r6, *(const int16_t*)( rsiLast ), 6 );
				r7 = _mm_insert_epi16( r7, *(const int16_t*)( rsiLast + sourceStride ), 7 );
				rsiLast += 2;

				// Bitwise operations are pretty fast, AMD Zen3 CPU can run 4 of them every clock cycle
				// Combine 8 vectors into one
				t0 = _mm_or_si128( r0, r1 );
				t1 = _mm_or_si128( r2, r3 );
				t2 = _mm_or_si128( r4, r5 );
				t3 = _mm_or_si128( r6, r7 );

				t4 = _mm_or_si128( t0, t1 );
				t5 = _mm_or_si128( t2, t3 );

				t6 = _mm_or_si128( t4, t5 );
				// Store 8 FP16 values, the destination is aligned
				_mm_store_si128( ( __m128i* )rdi, t6 );
				rdi += destStride;
			}
		}
	}

	__forceinline void transpose8PartialAvx2( uint16_t* rdiWords, size_t w, size_t h, const uint16_t* rsiWords, size_t sourceStride, size_t destStride )
	{
		assert( 0 == ( (size_t)rdiWords ) % 16 );
		assert( 0 == destStride % 8 );
		assert( w <= sourceStride );
		assert( h > 0 && h < 8 );

		// Scale strides to bytes, and cast the pointers
		sourceStride *= 2;
		destStride *= 2;
		uint8_t* rdi = (uint8_t*)rdiWords;
		const uint8_t* rsi = (const uint8_t*)rsiWords;

		const uint8_t* const rsiEndAligned = rsi + ( w & maskAlign16 ) * 2;
		const uint8_t* const rsiEnd = rsi + w * 2;
		const uint8_t* rsiMid = rsi + sourceStride * 3;
		const uint8_t* rsiLast = rsi + sourceStride * 6;
		uint8_t* rdiMid = rdi + destStride * 3;
		uint8_t* rdiLast = rdi + destStride * 6;

		while( rsi < rsiEndAligned )
		{
			// Load the block into 8 registers, set unused rows to zero
			__m256i r0 = load( rsi );
			__m256i r1 = _mm256_setzero_si256();
			__m256i r2 = _mm256_setzero_si256();
			__m256i r3 = _mm256_setzero_si256();
			__m256i r4 = _mm256_setzero_si256();
			__m256i r5 = _mm256_setzero_si256();
			__m256i r6 = _mm256_setzero_si256();
			// These branches, whether direct or indirect, are very predictable: same outcome for all iterations of the outer loop
			switch( h )
			{
			case 7:
				r6 = load( rsiLast );
			case 6:
				r5 = load( rsiMid + sourceStride * 2 );
			case 5:
				r4 = load( rsiMid + sourceStride );
			case 4:
				r3 = load( rsiMid );
			case 3:
				r2 = load( rsi + sourceStride * 2 );
			case 2:
				r1 = load( rsi + sourceStride );
			}
			rsi += 32;
			rsiMid += 32;
			rsiLast += 32;

			__m256i r7 = _mm256_setzero_si256();

			// Transpose FP16 values in registers
			TRANSPOSE_8X16();

			// Store
			STORE_8X16_LOW();

			STORE_8X16_HIGH();
		}

		if( rsi < rsiEnd )
		{
			// Loading 8 elements into corresponding lanes of 8 vectors
			// This way there's no data dependencies between these load instructions
			// Out of order execution should hopefully do it's magic in the CPU, running all these loads in parallel.
			__m128i r0;
			__m128i r1 = _mm_setzero_si128();
			__m128i r2 = _mm_setzero_si128();
			__m128i r3 = _mm_setzero_si128();
			__m128i r4 = _mm_setzero_si128();
			__m128i r5 = _mm_setzero_si128();
			__m128i r6 = _mm_setzero_si128();

			__m128i t0, t1, t2, t3, t4, t5;

#pragma loop( no_vector )
			while( rsi < rsiEnd )
			{
				r0 = _mm_cvtsi32_si128( *(const uint16_t*)rsi );

				switch( h )
				{
				case 7:
					r6 = _mm_insert_epi16( r6, *(const int16_t*)( rsiLast ), 6 );
				case 6:
					r5 = _mm_insert_epi16( r5, *(const int16_t*)( rsiMid + sourceStride * 2 ), 5 );
				case 5:
					r4 = _mm_insert_epi16( r4, *(const int16_t*)( rsiMid + sourceStride ), 4 );
				case 4:
					r3 = _mm_insert_epi16( r3, *(const int16_t*)( rsiMid ), 3 );
				case 3:
					r2 = _mm_insert_epi16( r2, *(const int16_t*)( rsi + sourceStride * 2 ), 2 );
				case 2:
					r1 = _mm_insert_epi16( r1, *(const int16_t*)( rsi + sourceStride ), 1 );
				}
				rsi += 2;
				rsiMid += 2;
				rsiLast += 2;

				// Bitwise operations are pretty fast, AMD Zen3 CPU can run 4 of them every clock cycle
				// Combine 7 vectors into one
				t0 = _mm_or_si128( r0, r1 );
				t1 = _mm_or_si128( r2, r3 );
				t2 = _mm_or_si128( r4, r5 );

				t3 = _mm_or_si128( t0, t1 );
				t4 = _mm_or_si128( t2, r6 );

				t5 = _mm_or_si128( t3, t4 );
				// Store 8 FP16 values, the destination is aligned
				_mm_store_si128( ( __m128i* )rdi, t5 );
				rdi += destStride;
			}
		}
	}
}

// At least for the hybrid decoder, this method absolutely dominates the CPU time.
// And not due to the integer shuffles - the bottleneck is loading data from the source matrix.
HRESULT MulMatBase::transposePanelAvx2( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const
{
	assert( stridesA[ 0 ] == 1 );

	const size_t heightFloats = (size_t)panelHeightRegisters * 8;
	i *= heightFloats;

	const uint16_t* rsi = (const uint16_t*)pa;
	rsi += m3 * stridesA[ 3 ];
	rsi += m2 * stridesA[ 2 ];
	rsi += i * stridesA[ 1 ];

	const size_t resultStride = heightFloats;

	if( i + heightFloats <= resultSize[ 0 ] )
	{
		// A complete panel
		for( size_t i = 0; i < panelHeightRegisters; i++ )
		{
			transpose8Avx2( rdi, length, rsi, stridesA[ 1 ], resultStride );
			// Advance by 8 floats in the output buffer
			rdi += 8;
			// Advance by 8 rows in the source matrix
			rsi += 8 * stridesA[ 1 ];
		}
	}
	else
	{
		// A partial panel, at the bottom of the first argument matrix
		const size_t remainder = resultSize[ 0 ] - i;
		assert( remainder > 0 && remainder < heightFloats );
		zeroAlignedMemory( rdi, resultStride * length * sizeof( uint16_t ) );

		const size_t completePanels = remainder / 8;
		for( size_t i = 0; i < completePanels; i++ )
		{
			transpose8Avx2( rdi, length, rsi, stridesA[ 1 ], resultStride );
			rdi += 8;
			rsi += 8 * stridesA[ 1 ];
		}
		const size_t lastPanel = remainder % 8;
		if( 0 != lastPanel )
			transpose8PartialAvx2( rdi, length, lastPanel, rsi, stridesA[ 1 ], resultStride );
	}
	return S_OK;
}