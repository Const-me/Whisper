#include "stdafx.h"
#include "miscUtils.h"
#include <cmath>

void setCurrentThreadName( const char* threadName )
{
	const DWORD dwThreadID = GetCurrentThreadId();

	// https://stackoverflow.com/a/10364541/126995
#pragma pack(push,8)
	typedef struct tagTHREADNAME_INFO
	{
		DWORD dwType;      // Must be 0x1000.
		LPCSTR szName;     // Pointer to name (in user addr space).
		DWORD dwThreadID;  // Thread ID (-1=caller thread).
		DWORD dwFlags;     // Reserved for future use, must be zero.
	} THREADNAME_INFO;
#pragma pack(pop)

	THREADNAME_INFO info;
	info.dwType = 0x1000;
	info.szName = threadName;
	info.dwThreadID = dwThreadID;
	info.dwFlags = 0;

	constexpr DWORD MS_VC_EXCEPTION = 0x406D1388;
	__try
	{
		RaiseException( MS_VC_EXCEPTION, 0, sizeof( info ) / sizeof( ULONG_PTR ), (ULONG_PTR*)&info );
	}
	__except( EXCEPTION_EXECUTE_HANDLER )
	{
	}
}

float computeScaling( int mul, int div )
{
#ifdef _DEBUG
	const float ref = (float)std::pow( (double)mul / (double)div, -0.25 );
#endif
	// Make int32 vector with both numbers
	__m128i iv = _mm_cvtsi32_si128( mul );
	iv = _mm_insert_epi32( iv, div, 1 );
	// Convert both numbers to FP64
	__m128d v = _mm_cvtepi32_pd( iv );
	// Compute mul / div
	v = _mm_div_sd( v, _mm_unpackhi_pd( v, v ) );
	// Square root
	v = _mm_sqrt_sd( v, v );
	// 4-th root
	v = _mm_sqrt_sd( v, v );
	// Invert the value
	v = _mm_div_sd( _mm_set_sd( 1.0 ), v );
	// Downcast to FP32, and return the result
	__m128 f32 = _mm_cvtsd_ss( _mm_setzero_ps(), v );
	return _mm_cvtss_f32( f32 );
}