#pragma once
#include <stdint.h>
#include <assert.h>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <atlstr.h>
#include <d3d11.h>

#include <vector>
#include <array>
#include <emmintrin.h>
#include <smmintrin.h>

#define CHECK( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) return __hr; }

inline __m128i load16( const int* rsi )
{
	return _mm_loadu_si128( ( const __m128i* )rsi );
}
inline __m128i load16( const uint32_t* rsi )
{
	return _mm_loadu_si128( ( const __m128i* )rsi );
}
inline __m128i load( const std::array<uint32_t, 4>& arr )
{
	return load16( arr.data() );
}

inline bool vectorEqual( __m128i a, __m128i b )
{
	__m128i xx = _mm_xor_si128( a, b );
	return (bool)_mm_testz_si128( xx, xx );
}

void printError( HRESULT hr );

inline const char* cstr( const CStringA& s ) { return s; }
inline const wchar_t* cstr( const CString& s ) { return s; }