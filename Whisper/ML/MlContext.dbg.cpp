#include "stdafx.h"
#include "MlContext.h"
#include "../source/ggml.h"
#include "testUtils.h"
using namespace DirectCompute;

#define E_TYPE HRESULT_FROM_WIN32( ERROR_DATATYPE_MISMATCH )

static void dbgPrintSizeDiff( const char* what, __m128i ref, __m128i gpu )
{
	std::array<int, 8> a;
	_mm_storeu_si128( ( __m128i* ) & a[ 0 ], ref );
	_mm_storeu_si128( ( __m128i* ) & a[ 4 ], gpu );
	printf( "%s; reference [ %i, %i, %i, %i ], GPGPU [ %i, %i, %i, %i ]\n",
		what,
		a[ 0 ], a[ 1 ], a[ 2 ], a[ 3 ],
		a[ 4 ], a[ 5 ], a[ 6 ], a[ 7 ] );
}

void MlContext::dbgPrintDifference( const ggml_tensor* reference, const Tensor& gpu, const char* what, bool trapToDebugger )
{
	sTensorDiff diff;
	const __m128i gpuSize = gpu.sizeVec();
	const __m128i gpuStrides = gpu.stridesVec();
	__m128i expectedStrides;
	if( reference->type == GGML_TYPE_F32 )
	{
		if( gpu.getType() != eDataType::FP32 )
			throw E_TYPE;
		expectedStrides = _mm_slli_epi32( gpuStrides, 2 );

		std::vector<float> v;
		gpu.download( v );
		diff = computeDiff( v.data(), (const float*)reference->data, v.size() );
	}
	else if( reference->type == GGML_TYPE_F16 )
	{
		if( gpu.getType() != eDataType::FP16 )
			throw E_TYPE;
		expectedStrides = _mm_slli_epi32( gpuStrides, 1 );

		std::vector<uint16_t> v;
		gpu.download( v );
		diff = computeDiff( v.data(), (const uint16_t*)reference->data, v.size() );
	}
	else
		throw E_NOTIMPL;

	const __m128i ggmlSize = _mm_loadu_si128( ( const __m128i* ) & reference->ne[ 0 ] );
	const __m128i ggmlStrides = _mm_loadu_si128( ( const __m128i* ) & reference->nb[ 0 ] );
	if( !vectorEqual( gpuSize, ggmlSize ) )
		dbgPrintSizeDiff( "dbgPrintDifference - size is different", ggmlSize, gpuSize );
	// if( !vectorEqual( expectedStrides, ggmlStrides ) ) dbgPrintSizeDiff( "dbgPrintDifference - stride is different", ggmlStrides, expectedStrides );

	diff.print( what );

	if( trapToDebugger )
		__debugbreak();
}