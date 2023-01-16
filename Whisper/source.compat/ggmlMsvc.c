#include <stdint.h>
#include <immintrin.h>
#include <stdio.h>
#include <assert.h>
#include "../source/ggml.h"

__forceinline float _cvtsh_ss( uint16_t f16 )
{
	__m128i i = _mm_cvtsi32_si128( f16 );
	__m128 f = _mm_cvtph_ps( i );
	return _mm_cvtss_f32( f );
}

__forceinline uint16_t _cvtss_sh( float f, int rounding )
{
	assert( 0 == rounding );
	__m128 v = _mm_set_ss( f );
	__m128i i = _mm_cvtps_ph( v, 0 );
	return (uint16_t)(uint32_t)_mm_cvtsi128_si32( i );
}

FILE* fopen_msvc( const char* filename, const char* mode )
{
	FILE* stream;
	errno_t err = fopen_s( &stream, filename, mode );
	if( err == 0 )
		return stream;
	return NULL;
}

#define fopen fopen_msvc

#include "../ML/testUtilsC.h"

#define __F16C__
#define __FMA__
#include "../source/ggml.c"