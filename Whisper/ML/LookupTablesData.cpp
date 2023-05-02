#include "stdafx.h"
#include "LookupTablesData.h"
#include <immintrin.h>
#include <atlfile.h>
#include <Utils/LZ4/lz4.h>
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

#ifndef __AVX__
namespace
{
	// Compressed data of these two lookup tables
#include "LookupTablesData.inl"
}
#endif

LookupTablesData::LookupTablesData()
{
#ifdef __AVX__
	// When compiling for AVX, we assume the CPU also has F16C
	// https://en.wikipedia.org/wiki/F16C
	for( int i = 0; i < 0x10000; i++ )
	{
		const float f = fp32( i );
		gelu[ i ] = fp16( computeGelu( f ) );
		exponent[ i ] = fp16( (float)exp( f ) );
	}
#else
	// When compiling without AVX, use the data compiled into the DLL
	constexpr int cbThis = (int)( sizeof( *this ) );
	const int lz4Status = LZ4_decompress_safe( (const char*)s_tableData.data(), (char*)this, (int)s_tableData.size(), cbThis );
	if( lz4Status != cbThis )
	{
		logError( u8"LZ4_decompress_safe failed with status %i", lz4Status );
		throw PLA_E_CABAPI_FAILURE;
	}
#endif

#if false
	// Temporary code to save the content of these lookup tables
	CAtlFile tempFile;
	tempFile.Create( LR"(C:\Temp\2remove\Whisper\tables.bin)", GENERIC_WRITE, 0, CREATE_ALWAYS );
	tempFile.Write( this, (DWORD)sizeof( *this ) );
#endif
}