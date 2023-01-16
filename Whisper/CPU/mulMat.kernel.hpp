#pragma once
#include <stdint.h>
#include <immintrin.h>
#include "simdUtils.h"

template<uint8_t panelHeightRegs, uint8_t tileWidthFloats>
struct ResultTile
{
	static constexpr size_t totalRegs = (size_t)(tileWidthFloats)*panelHeightRegs;
	std::array<__m256, totalRegs> arr;

	template<size_t idx>
	__forceinline void fmadd( __m256 a, __m256 b )
	{
		arr[ idx ] = _mm256_fmadd_ps( a, b, arr[ idx ] );
	}
	__forceinline void kernel( const std::array<__m256, panelHeightRegs>& panel, const float* rsi, size_t stride );
	__forceinline void kernelPartial( const std::array<__m256, panelHeightRegs>& panel, const float* rsi, size_t stride, size_t rem )
	{
		throw E_UNEXPECTED;
	}
	__forceinline void store( float* rdi, size_t w, size_t h, size_t stride ) const;
};

#pragma region setZero functions
__forceinline void setZero( std::array<__m256, 1>& dest )
{
	dest[ 0 ] = _mm256_setzero_ps();
}
__forceinline void setZero( std::array<__m256, 2>& dest )
{
	dest[ 0 ] = _mm256_setzero_ps();
	dest[ 1 ] = _mm256_setzero_ps();
}
__forceinline void setZero( std::array<__m256, 3>& dest )
{
	dest[ 0 ] = _mm256_setzero_ps();
	dest[ 1 ] = _mm256_setzero_ps();
	dest[ 2 ] = _mm256_setzero_ps();
}
__forceinline void setZero( std::array<__m256, 4>& dest )
{
	dest[ 0 ] = _mm256_setzero_ps();
	dest[ 1 ] = _mm256_setzero_ps();
	dest[ 2 ] = _mm256_setzero_ps();
	dest[ 3 ] = _mm256_setzero_ps();
}
__forceinline void setZero( std::array<__m256, 6>& dest )
{
	dest[ 0 ] = _mm256_setzero_ps();
	dest[ 1 ] = _mm256_setzero_ps();
	dest[ 2 ] = _mm256_setzero_ps();
	dest[ 3 ] = _mm256_setzero_ps();
	dest[ 4 ] = _mm256_setzero_ps();
	dest[ 5 ] = _mm256_setzero_ps();
}
__forceinline void setZero( std::array<__m256, 8>& dest )
{
	dest[ 0 ] = _mm256_setzero_ps();
	dest[ 1 ] = _mm256_setzero_ps();
	dest[ 2 ] = _mm256_setzero_ps();
	dest[ 3 ] = _mm256_setzero_ps();
	dest[ 4 ] = _mm256_setzero_ps();
	dest[ 5 ] = _mm256_setzero_ps();
	dest[ 6 ] = _mm256_setzero_ps();
	dest[ 7 ] = _mm256_setzero_ps();
}
#pragma endregion

#pragma region Micro-kernels
__forceinline void ResultTile<1, 1>::kernel( const std::array<__m256, 1>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
}
__forceinline void ResultTile<1, 2>::kernel( const std::array<__m256, 1>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	b = _mm256_broadcast_ss( rsi + stride );
	fmadd<1>( panel[ 0 ], b );
}
__forceinline void ResultTile<1, 2>::kernelPartial( const std::array<__m256, 1>& panel, const float* rsi, size_t stride, size_t rem )
{
	assert( 1 == rem );
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
}
__forceinline void ResultTile<1, 3>::kernel( const std::array<__m256, 1>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	b = _mm256_broadcast_ss( rsi + stride );
	fmadd<1>( panel[ 0 ], b );
	b = _mm256_broadcast_ss( rsi + stride * 2 );
	fmadd<2>( panel[ 0 ], b );
}
__forceinline void ResultTile<1, 3>::kernelPartial( const std::array<__m256, 1>& panel, const float* rsi, size_t stride, size_t rem )
{
	assert( rem > 0 && rem < 3 );
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	if( rem > 1 )
	{
		b = _mm256_broadcast_ss( rsi + stride );
		fmadd<1>( panel[ 0 ], b );
	}
}

__forceinline void ResultTile<1, 4>::kernel( const std::array<__m256, 1>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	b = _mm256_broadcast_ss( rsi + stride );
	fmadd<1>( panel[ 0 ], b );
	b = _mm256_broadcast_ss( rsi + stride * 2 );
	fmadd<2>( panel[ 0 ], b );
	b = _mm256_broadcast_ss( rsi + stride * 3 );
	fmadd<3>( panel[ 0 ], b );
}
__forceinline void ResultTile<1, 4>::kernelPartial( const std::array<__m256, 1>& panel, const float* rsi, size_t stride, size_t rem )
{
	assert( rem > 0 && rem < 4 );
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );

	switch( rem )
	{
	case 3:
		b = _mm256_broadcast_ss( rsi + stride * 2 );
		fmadd<2>( panel[ 0 ], b );
	case 2:
		b = _mm256_broadcast_ss( rsi + stride );
		fmadd<1>( panel[ 0 ], b );
	}
}
__forceinline void ResultTile<4, 1>::kernel( const std::array<__m256, 4>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	fmadd<1>( panel[ 1 ], b );
	fmadd<2>( panel[ 2 ], b );
	fmadd<3>( panel[ 3 ], b );
}
__forceinline void ResultTile<2, 4>::kernel( const std::array<__m256, 2>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	fmadd<1>( panel[ 1 ], b );

	b = _mm256_broadcast_ss( rsi + stride );
	fmadd<2>( panel[ 0 ], b );
	fmadd<3>( panel[ 1 ], b );

	b = _mm256_broadcast_ss( rsi + stride * 2 );
	fmadd<4>( panel[ 0 ], b );
	fmadd<5>( panel[ 1 ], b );

	b = _mm256_broadcast_ss( rsi + stride * 3 );
	fmadd<6>( panel[ 0 ], b );
	fmadd<7>( panel[ 1 ], b );
}

__forceinline void ResultTile<2, 4>::kernelPartial( const std::array<__m256, 2>& panel, const float* rsi, size_t stride, size_t rem )
{
	assert( rem > 0 && rem < 4 );
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	fmadd<1>( panel[ 1 ], b );

	switch( rem )
	{
	case 3:
		b = _mm256_broadcast_ss( rsi + stride * 2 );
		fmadd<4>( panel[ 0 ], b );
		fmadd<5>( panel[ 1 ], b );
	case 2:
		b = _mm256_broadcast_ss( rsi + stride );
		fmadd<2>( panel[ 0 ], b );
		fmadd<3>( panel[ 1 ], b );
	}
}

__forceinline void ResultTile<2, 3>::kernel( const std::array<__m256, 2>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	fmadd<1>( panel[ 1 ], b );

	b = _mm256_broadcast_ss( rsi + stride );
	fmadd<2>( panel[ 0 ], b );
	fmadd<3>( panel[ 1 ], b );

	b = _mm256_broadcast_ss( rsi + stride * 2 );
	fmadd<4>( panel[ 0 ], b );
	fmadd<5>( panel[ 1 ], b );
}
__forceinline void ResultTile<2, 3>::kernelPartial( const std::array<__m256, 2>& panel, const float* rsi, size_t stride, size_t rem )
{
	assert( rem > 0 && rem < 3 );
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	fmadd<1>( panel[ 1 ], b );
	if( rem > 1 )
	{
		b = _mm256_broadcast_ss( rsi + stride );
		fmadd<2>( panel[ 0 ], b );
		fmadd<3>( panel[ 1 ], b );
	}
}

__forceinline void ResultTile<4, 2>::kernel( const std::array<__m256, 4>& panel, const float* rsi, size_t stride )
{
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	fmadd<1>( panel[ 1 ], b );
	fmadd<2>( panel[ 2 ], b );
	fmadd<3>( panel[ 3 ], b );

	b = _mm256_broadcast_ss( rsi + stride );
	fmadd<4>( panel[ 0 ], b );
	fmadd<5>( panel[ 1 ], b );
	fmadd<6>( panel[ 2 ], b );
	fmadd<7>( panel[ 3 ], b );
}
__forceinline void ResultTile<4, 2>::kernelPartial( const std::array<__m256, 4>& panel, const float* rsi, size_t stride, size_t rem )
{
	assert( 1 == rem );
	__m256 b = _mm256_broadcast_ss( rsi );
	fmadd<0>( panel[ 0 ], b );
	fmadd<1>( panel[ 1 ], b );
	fmadd<2>( panel[ 2 ], b );
	fmadd<3>( panel[ 3 ], b );
}
#pragma endregion

#pragma region Loads
// This function should compile into a single `vcvtph2ps` instruction, with memory operand
__forceinline __m256 loadUpcasted( const uint16_t* rsi )
{
	__m128i i = _mm_load_si128( ( const __m128i* )rsi );
	return _mm256_cvtph_ps( i );
}

// We loading the panel from the temporary buffer.
// For this reason, we don't need to handle remainders, the code which made the buffer wrote zeros into the remainder elements
// We can even use aligned load instructions.
__forceinline void loadPanel( const uint16_t* rsi, std::array<__m256, 1>& dest )
{
	dest[ 0 ] = loadUpcasted( rsi );
}
__forceinline void loadPanel( const uint16_t* rsi, std::array<__m256, 2>& dest )
{
	dest[ 0 ] = loadUpcasted( rsi );
	dest[ 1 ] = loadUpcasted( rsi + 8 );
}
__forceinline void loadPanel( const uint16_t* rsi, std::array<__m256, 3>& dest )
{
	dest[ 0 ] = loadUpcasted( rsi );
	dest[ 1 ] = loadUpcasted( rsi + 8 );
	dest[ 2 ] = loadUpcasted( rsi + 8 * 2 );
}
__forceinline void loadPanel( const uint16_t* rsi, std::array<__m256, 4>& dest )
{
	dest[ 0 ] = loadUpcasted( rsi );
	dest[ 1 ] = loadUpcasted( rsi + 8 );
	dest[ 2 ] = loadUpcasted( rsi + 8 * 2 );
	dest[ 3 ] = loadUpcasted( rsi + 8 * 3 );
}
#pragma endregion

#pragma region Stores
__forceinline void ResultTile<1, 1>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h == 1 && w > 0 && w <= 8 );
	if( w == 8 )
		_mm256_storeu_ps( rdi, arr[ 0 ] );
	else
	{
		const __m256i mask = loadTailMaskInt( w );
		_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
	}
}

__forceinline void ResultTile<1, 2>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h > 0 && w > 0 && h <= 2 && w <= 8 );
	if( w == 8 )
	{
		switch( h )
		{
		case 2:
			_mm256_storeu_ps( rdi + stride, arr[ 1 ] );
		case 1:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
		}
	}
	else
	{
		const __m256i mask = loadTailMaskInt( w );
		switch( h )
		{
		case 2:
			_mm256_maskstore_ps( rdi + stride, mask, arr[ 1 ] );
		case 1:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
		}
	}
}

__forceinline void ResultTile<1, 3>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h > 0 && w > 0 && h <= 3 && w <= 8 );
	if( w == 8 )
	{
		switch( h )
		{
		case 3:
			_mm256_storeu_ps( rdi + stride * 2, arr[ 2 ] );
		case 2:
			_mm256_storeu_ps( rdi + stride, arr[ 1 ] );
		case 1:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
		}
	}
	else
	{
		const __m256i mask = loadTailMaskInt( w );
		switch( h )
		{
		case 3:
			_mm256_maskstore_ps( rdi + stride * 2, mask, arr[ 2 ] );
		case 2:
			_mm256_maskstore_ps( rdi + stride, mask, arr[ 1 ] );
		case 1:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
		}
	}
}

__forceinline void ResultTile<1, 4>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h > 0 && w > 0 && h <= 4 && w <= 8 );

	if( w == 8 )
	{
		switch( h )
		{
		case 4:
			_mm256_storeu_ps( rdi + stride * 3, arr[ 3 ] );
		case 3:
			_mm256_storeu_ps( rdi + stride * 2, arr[ 2 ] );
		case 2:
			_mm256_storeu_ps( rdi + stride, arr[ 1 ] );
		case 1:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
		}
	}
	else
	{
		const __m256i mask = loadTailMaskInt( w );
		switch( h )
		{
		case 4:
			_mm256_maskstore_ps( rdi + stride * 3, mask, arr[ 3 ] );
		case 3:
			_mm256_maskstore_ps( rdi + stride * 2, mask, arr[ 2 ] );
		case 2:
			_mm256_maskstore_ps( rdi + stride, mask, arr[ 1 ] );
		case 1:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
		}
	}
}

__forceinline void ResultTile<4, 1>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h == 1 && w > 0 && w <= 32 );
	if( w == 32 )
	{
		// 4 complete vectors, this branch is very likely to be taken
		_mm256_storeu_ps( rdi, arr[ 0 ] );
		_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
		_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );
		_mm256_storeu_ps( rdi + 8 * 3, arr[ 3 ] );
	}
	else
	{
		const size_t rem = w % 8;
		const __m256i mask = loadTailMaskInt<false>( rem );
		const size_t completeVectors = w / 8;
		const size_t key = ( completeVectors << 1 ) | ( ( 0 == rem ) ? 0 : 1 );
		switch( key )
		{
		case 1:	// 0 complete vectors + remainder
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			break;
		case 2:	// 1 complete vector
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			break;
		case 3:	// 1 complete vector + remainder
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			break;
		case 4:	// 2 complete vectors
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			break;
		case 5:	// 2 complete vectors + remainder
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_maskstore_ps( rdi + 8 * 2, mask, arr[ 2 ] );
			break;
		case 6:	// 3 complete vectors
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );
			break;
		case 7:	// 3 complete vectors + remainder
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );
			_mm256_maskstore_ps( rdi + 8 * 3, mask, arr[ 3 ] );
			break;
		default:
			throw E_UNEXPECTED;
		}
	}
}
__forceinline void ResultTile<4, 2>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h > 0 && w > 0 && h <= 2 && w <= 32 );
	const bool twoRows = h == 2;
	float* const rdi1 = rdi + stride;
	if( w == 32 )
	{
		_mm256_storeu_ps( rdi, arr[ 0 ] );
		_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
		_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );
		_mm256_storeu_ps( rdi + 8 * 3, arr[ 3 ] );

		if( twoRows )
		{
			_mm256_storeu_ps( rdi1, arr[ 4 ] );
			_mm256_storeu_ps( rdi1 + 8, arr[ 5 ] );
			_mm256_storeu_ps( rdi1 + 8 * 2, arr[ 6 ] );
			_mm256_storeu_ps( rdi1 + 8 * 3, arr[ 7 ] );
		}
	}
	else
	{
		const size_t rem = w % 8;
		const __m256i mask = loadTailMaskInt<false>( rem );
		const size_t completeVectors = w / 8;
		// Lowest bit: remainder
		// Next bit: set when storing 2 rows
		// Next 2 bits: count of complete vectors in X direction, [ 0..3 ]
		const size_t key = ( completeVectors << 2 ) | ( ( 0 == rem ) ? 0 : 1 ) | ( twoRows ? 2 : 0 );
		switch( key )
		{
		case 1:	// 0 complete vectors + remainder, 1 row
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			break;
		case 3:	// 0 complete vectors + remainder, 2 rows
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			_mm256_maskstore_ps( rdi1, mask, arr[ 4 ] );
			break;
		case 4:	// 1 complete vector, 1 row
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			break;
		case 5:	// 1 complete vector + remainder, 1 row
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			break;
		case 6:	// 1 complete vector, 2 rows
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi1, arr[ 4 ] );
			break;
		case 7:	// 1 complete vector + remainder, 2 rows
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );

			_mm256_storeu_ps( rdi1, arr[ 4 ] );
			_mm256_maskstore_ps( rdi1 + 8, mask, arr[ 5 ] );
			break;
		case 8:	// 2 complete vectors, 1 row
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			break;
		case 9:	// 2 complete vectors + remainder, 1 row
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_maskstore_ps( rdi + 8 * 2, mask, arr[ 2 ] );
			break;
		case 10:	// 2 complete vectors, 2 rows
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );

			_mm256_storeu_ps( rdi1, arr[ 4 ] );
			_mm256_storeu_ps( rdi1 + 8, arr[ 5 ] );
			break;
		case 11:	// 2 complete vectors + remainder, 2 rows
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_maskstore_ps( rdi + 8 * 2, mask, arr[ 2 ] );

			_mm256_storeu_ps( rdi1, arr[ 4 ] );
			_mm256_storeu_ps( rdi1 + 8, arr[ 5 ] );
			_mm256_maskstore_ps( rdi1 + 8 * 2, mask, arr[ 6 ] );
			break;
		case 12:	// 3 complete vectors, 1 row
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );
			break;
		case 13:	// 3 complete vectors + remainder, 1 row
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );
			_mm256_maskstore_ps( rdi + 8 * 3, mask, arr[ 3 ] );
			break;
		case 14:	// 3 complete vectors, 2 rows
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );

			_mm256_storeu_ps( rdi1, arr[ 4 ] );
			_mm256_storeu_ps( rdi1 + 8, arr[ 5 ] );
			_mm256_storeu_ps( rdi1 + 8 * 2, arr[ 6 ] );
			break;
		case 15:	// 3 complete vectors + remainder, 2 rows
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
			_mm256_storeu_ps( rdi + 8 * 2, arr[ 2 ] );
			_mm256_maskstore_ps( rdi + 8 * 3, mask, arr[ 3 ] );

			_mm256_storeu_ps( rdi1, arr[ 4 ] );
			_mm256_storeu_ps( rdi1 + 8, arr[ 5 ] );
			_mm256_storeu_ps( rdi1 + 8 * 2, arr[ 6 ] );
			_mm256_maskstore_ps( rdi1 + 8 * 3, mask, arr[ 7 ] );
			break;
		default:
			throw E_UNEXPECTED;
		}
	}
}

__forceinline void ResultTile<2, 4>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h > 0 && w > 0 && h <= 4 && w <= 16 );
	h--;
	float* const rdi1 = rdi + stride;
	float* const rdi2 = rdi + stride * 2;
	float* const rdi3 = rdi + stride * 3;

	if( w == 16 )
	{
		switch( h )
		{
		case 3:
			_mm256_storeu_ps( rdi3, arr[ 6 ] );
			_mm256_storeu_ps( rdi3 + 8, arr[ 7 ] );
		case 2:
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			_mm256_storeu_ps( rdi2 + 8, arr[ 5 ] );
		case 1:
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_storeu_ps( rdi1 + 8, arr[ 3 ] );
		case 0:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
		}
	}
	else
	{
		const size_t rem = w % 8;
		const __m256i mask = loadTailMaskInt<false>( rem );
		// 0 for partial first vector, 1 for exactly 1 complete vector, 2 for 1 complete vector with remainder
		const size_t partialCase = ( w < 8 ) ? 0 : ( ( w == 8 ) ? 1 : 2 );
		// Merge into a single integer for the switch statement
		const size_t key = partialCase + h * 3;

		switch( key )
		{
			// h = 1
		case 0:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			break;
		case 1:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			break;
		case 2:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			break;
			// h = 2
		case 3:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			_mm256_maskstore_ps( rdi1, mask, arr[ 2 ] );
			break;
		case 4:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			break;
		case 5:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_maskstore_ps( rdi1 + 8, mask, arr[ 3 ] );
			break;
			// h = 3
		case 6:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			_mm256_maskstore_ps( rdi1, mask, arr[ 2 ] );
			_mm256_maskstore_ps( rdi2, mask, arr[ 4 ] );
			break;
		case 7:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			break;
		case 8:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_maskstore_ps( rdi1 + 8, mask, arr[ 3 ] );
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			_mm256_maskstore_ps( rdi2 + 8, mask, arr[ 5 ] );
			break;
			// h = 4
		case 9:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			_mm256_maskstore_ps( rdi1, mask, arr[ 2 ] );
			_mm256_maskstore_ps( rdi2, mask, arr[ 4 ] );
			_mm256_maskstore_ps( rdi3, mask, arr[ 6 ] );
			break;
		case 10:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			_mm256_storeu_ps( rdi3, arr[ 6 ] );
			break;
		case 11:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_maskstore_ps( rdi1 + 8, mask, arr[ 3 ] );
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			_mm256_maskstore_ps( rdi2 + 8, mask, arr[ 5 ] );
			_mm256_storeu_ps( rdi3, arr[ 6 ] );
			_mm256_maskstore_ps( rdi3 + 8, mask, arr[ 7 ] );
			break;
		default:
			throw E_UNEXPECTED;
		}
	}
}

__forceinline void ResultTile<2, 3>::store( float* rdi, size_t w, size_t h, size_t stride ) const
{
	assert( h > 0 && w > 0 && h <= 3 && w <= 16 );
	float* const rdi1 = rdi + stride;
	float* const rdi2 = rdi + stride * 2;
	h--;

	if( w == 16 )
	{
		switch( h )
		{
		case 2:
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			_mm256_storeu_ps( rdi2 + 8, arr[ 5 ] );
		case 1:
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_storeu_ps( rdi1 + 8, arr[ 3 ] );
		case 0:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi + 8, arr[ 1 ] );
		}
	}
	else
	{
		const size_t rem = w % 8;
		const __m256i mask = loadTailMaskInt<false>( rem );
		// 0 for partial first vector, 1 for exactly 1 complete vector, 2 for 1 complete vector with remainder
		const size_t partialCase = ( w < 8 ) ? 0 : ( ( w == 8 ) ? 1 : 2 );
		// Merge into a single integer for the switch statement
		const size_t key = partialCase + h * 3;

		switch( key )
		{
			// h = 1
		case 0:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			break;
		case 1:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			break;
		case 2:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			break;
			// h = 2
		case 3:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			_mm256_maskstore_ps( rdi1, mask, arr[ 2 ] );
			break;
		case 4:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			break;
		case 5:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_maskstore_ps( rdi1 + 8, mask, arr[ 3 ] );
			break;
			// h = 3
		case 6:
			_mm256_maskstore_ps( rdi, mask, arr[ 0 ] );
			_mm256_maskstore_ps( rdi1, mask, arr[ 2 ] );
			_mm256_maskstore_ps( rdi2, mask, arr[ 4 ] );
			break;
		case 7:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			break;
		case 8:
			_mm256_storeu_ps( rdi, arr[ 0 ] );
			_mm256_maskstore_ps( rdi + 8, mask, arr[ 1 ] );
			_mm256_storeu_ps( rdi1, arr[ 2 ] );
			_mm256_maskstore_ps( rdi1 + 8, mask, arr[ 3 ] );
			_mm256_storeu_ps( rdi2, arr[ 4 ] );
			_mm256_maskstore_ps( rdi2 + 8, mask, arr[ 5 ] );
			break;
		default:
			throw E_UNEXPECTED;
		}
	}
}
#pragma endregion