#include "stdafx.h"
#include <intrin.h>
#include "mulMatImpl.h"
#include "mulMatUtils.hpp"
using namespace CpuCompute;

// We want to keep code size reasonable, that's why these panel reshaping methods are in the base class
HRESULT MulMatBase::transposePanel( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const
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
			transpose8( rdi, length, rsi, stridesA[ 1 ], resultStride );
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
			transpose8( rdi, length, rsi, stridesA[ 1 ], resultStride );
			rdi += 8;
			rsi += 8 * stridesA[ 1 ];
		}
		const size_t lastPanel = remainder % 8;
		if( 0 != lastPanel )
			transpose8Partial( rdi, length, lastPanel, rsi, stridesA[ 1 ], resultStride );
	}
	return S_OK;
}

inline const uint16_t* MulMatBase::getPanelA( size_t i, size_t m2, size_t m3 ) const
{
	const uint16_t* rsi = (const uint16_t*)pa;
	rsi += m3 * stridesA[ 3 ];
	rsi += m2 * stridesA[ 2 ];
	rsi += i * stridesA[ 1 ];
	return rsi;
}

HRESULT MulMatBase::copyPanelColumnMajor8( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const
{
	assert( stridesA[ 1 ] == 1 );
	assert( panelHeightRegisters == 1 );

	constexpr size_t heightFloats = 8;
	i *= heightFloats;
	const uint16_t* rsi = getPanelA( i, m2, m3 );

	constexpr size_t resultStride = heightFloats;

	if( i + heightFloats <= resultSize[ 0 ] )
	{
		// A complete panel, height = 8 elements
		copyColumnMajor( rdi, length, rsi, stridesA[ 0 ], resultStride );
	}
	else
	{
		// A partial panel, at the bottom of the first argument matrix
		const size_t remainder = resultSize[ 0 ] - i;
		assert( remainder > 0 && remainder < heightFloats );
		copyColumnMajorPartial( rdi, length, remainder, rsi, stridesA[ 0 ], resultStride );
	}
	return S_OK;
}

__forceinline __m128i load8Partial( const uint16_t* x, size_t len )
{
	assert( len > 0 && len < 8 );
	__m128i ix = _mm_setzero_si128();
	switch( len )
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
	}
	return ix;
}

__forceinline __m256i load16Partial( const uint16_t* rsi, size_t len )
{
	assert( len > 0 && len < 16 );

	if( len < 8 )
	{
		__m128i low = load8Partial( rsi, len );
		return _mm256_setr_m128i( low, _mm_setzero_si128() );
	}
	else if( len > 8 )
	{
		__m128i low = load16( (const int*)rsi );
		__m128i high = load8Partial( rsi + 8, len - 8 );
		return _mm256_setr_m128i( low, high );
	}
	else
	{
		__m128i low = load16( (const int*)rsi );
		return _mm256_setr_m128i( low, _mm_setzero_si128() );
	}
}

HRESULT MulMatBase::copyPanelColumnMajor16( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const
{
	assert( stridesA[ 1 ] == 1 );
	assert( panelHeightRegisters == 2 );

	constexpr size_t heightFloats = 16;
	i *= heightFloats;

	const uint16_t* rsi = getPanelA( i, m2, m3 );
	uint16_t* const rdiEnd = rdi + 16 * length;

	if( i + heightFloats <= resultSize[ 0 ] )
	{
		// A complete panel, height = 16 elements
		for( ; rdi < rdiEnd; rdi += 16, rsi += stridesA[ 0 ] )
		{
			__m256i v = _mm256_loadu_si256( ( const __m256i* )rsi );
			_mm256_store_si256( ( __m256i* )rdi, v );
		}
	}
	else
	{
		// A partial panel, at the bottom of the first argument matrix
		const size_t remainder = resultSize[ 0 ] - i;
		assert( remainder > 0 && remainder < heightFloats );

		for( ; rdi < rdiEnd; rdi += 16, rsi += stridesA[ 0 ] )
		{
			__m256i v = load16Partial( rsi, remainder );
			_mm256_store_si256( ( __m256i* )rdi, v );
		}
	}
	return S_OK;
}

HRESULT MulMatBase::copyPanelColumnMajor32( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const
{
	assert( stridesA[ 1 ] == 1 );
	assert( panelHeightRegisters == 4 );

	constexpr size_t heightFloats = 32;
	i *= heightFloats;

	const uint16_t* rsi = getPanelA( i, m2, m3 );
	uint16_t* const rdiEnd = rdi + 32 * length;

	if( i + heightFloats <= resultSize[ 0 ] )
	{
		// A complete panel, height = 32 elements
		for( ; rdi < rdiEnd; rdi += 32, rsi += stridesA[ 0 ] )
		{
			__m256i v = _mm256_loadu_si256( ( const __m256i* )rsi );
			_mm256_store_si256( ( __m256i* )rdi, v );
			v = _mm256_loadu_si256( ( const __m256i* )( rsi + 16 ) );
			_mm256_store_si256( ( __m256i* )( rdi + 16 ), v );
		}
	}
	else
	{
		// A partial panel, at the bottom of the first argument matrix
		const size_t remainder = resultSize[ 0 ] - i;
		assert( remainder > 0 && remainder < heightFloats );

		// _mm256_setzero_si256 probably compiles into vpxor, that's AVX2, we don't want that here
		const __m256 zero = _mm256_setzero_ps();

		for( ; rdi < rdiEnd; rdi += 32, rsi += stridesA[ 0 ] )
		{
			if( remainder < 16 )
			{
				__m256i v = load16Partial( rsi, remainder );
				_mm256_store_si256( ( __m256i* )rdi, v );
				_mm256_store_ps( (float*)( rdi + 16 ), zero );
			}
			else if( remainder > 16 )
			{
				__m256i v = _mm256_loadu_si256( ( const __m256i* )rsi );
				_mm256_store_si256( ( __m256i* )rdi, v );
				v = load16Partial( rsi + 16, remainder - 16 );
				_mm256_store_si256( ( __m256i* )( rdi + 16 ), v );
			}
			else
			{
				__m256i v = _mm256_loadu_si256( ( const __m256i* )rsi );
				_mm256_store_si256( ( __m256i* )rdi, v );
				_mm256_store_ps( (float*)( rdi + 16 ), zero );
			}
		}
	}
	return S_OK;
}

HRESULT MulMatBase::gatherPanel( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const
{
	// BTW, I never saw this method called.
	const size_t heightFloats = (size_t)panelHeightRegisters * 8;
	const size_t length = this->length;

	zeroAlignedMemory( rdi, length * heightFloats * sizeof( uint16_t ) );

	const size_t height = std::min( heightFloats, resultSize[ 0 ] - i );
	const size_t strideElement = stridesA[ 0 ];
	const size_t strideRow = stridesA[ 1 ];
	const uint16_t* rsi = getPanelA( i * heightFloats, m2, m3 );

	if( strideElement < strideRow )
	{
		for( size_t r = 0; r < height; r++, rsi += strideRow, rdi++ )
		{
			const uint16_t* sourceRow = rsi;
			uint16_t* destRow = rdi;
			for( size_t c = 0; c < length; c++, sourceRow += strideElement, destRow += heightFloats )
				*destRow = *sourceRow;
		}
	}
	else
	{
		for( size_t c = 0; c < length; c++, rsi += strideElement, rdi += heightFloats )
		{
			const uint16_t* sourceCol = rsi;
			uint16_t* destCol = rdi;
			for( size_t r = 0; r < height; r++, sourceCol += strideRow, destCol++ )
				*destCol = *sourceCol;
		}
	}
	return S_OK;
}