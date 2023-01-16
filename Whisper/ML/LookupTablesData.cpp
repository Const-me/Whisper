#include "stdafx.h"
#include "LookupTablesData.h"
#include <immintrin.h>
using namespace DirectCompute;

namespace
{
	inline float fp32( uint16_t f16 )
	{
		__m128i i = _mm_cvtsi32_si128( f16 );
		__m128 f = _mm_cvtph_ps( i );
		return _mm_cvtss_f32( f );
	}

	inline uint16_t fp16( float fp32 )
	{
		__m128 f = _mm_set_ss( fp32 );
		__m128i i = _mm_cvtps_ph( f, 0 );
		uint32_t res = (uint32_t)_mm_cvtsi128_si32( i );
		return (uint16_t)res;
	}

	constexpr double GELU_COEF_A = 0.044715;
	constexpr double SQRT_2_OVER_PI = 0.79788456080286535587989211986876;

	inline float computeGelu( float x )
	{
		return (float)( 0.5 * x * ( 1.0 + tanh( SQRT_2_OVER_PI * x * ( 1.0 + GELU_COEF_A * x * x ) ) ) );
	}
}

LookupTablesData::LookupTablesData()
{
	for( int i = 0; i < 0x10000; i++ )
	{
		const float f = fp32( i );
		gelu[ i ] = fp16( computeGelu( f ) );
		exponent[ i ] = fp16( (float)exp( f ) );
	}
}