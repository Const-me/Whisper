#include "stdafx.h"
#include "TensorShape.h"
#include "../source/ggml.h"
using namespace DirectCompute;

TensorShape::TensorShape()
{
	setZero();
}

TensorShape::TensorShape( const TensorShape& that )
{
	_mm_storeu_si128( ( __m128i* )ne.data(), that.sizeVec() );
	_mm_storeu_si128( ( __m128i* )nb.data(), that.stridesVec() );
}

void TensorShape::operator=( const TensorShape& that )
{
	_mm_storeu_si128( ( __m128i* )ne.data(), that.sizeVec() );
	_mm_storeu_si128( ( __m128i* )nb.data(), that.stridesVec() );
}

void TensorShape::setDenseStrides()
{
	nb[ 0 ] = 1;
	nb[ 1 ] = ne[ 0 ];
	const uint32_t p01 = ne[ 0 ] * ne[ 1 ];
	nb[ 2 ] = p01;
	nb[ 3 ] = p01 * ne[ 2 ];
}

bool DirectCompute::canMulMat( const TensorShape& t0, const TensorShape& t1 )
{
	/*
	return
		( t0.ne[ 0 ] == t1.ne[ 0 ] ) &&
		( t0.ne[ 2 ] == t1.ne[ 2 ] ) &&
		( t0.ne[ 3 ] == t1.ne[ 3 ] ); */
	__m128i a = t0.sizeVec();
	__m128i b = t1.sizeVec();
	__m128i xx = _mm_xor_si128( a, b );
	xx = _mm_shuffle_epi32( xx, _MM_SHUFFLE( 3, 2, 0, 0 ) );
	return (bool)_mm_testz_si128( xx, xx );
}