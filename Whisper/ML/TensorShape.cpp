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

HRESULT TensorShape::create( const ggml_tensor& ggml )
{
	for( size_t i = 0; i < 4; i++ )
		ne[ i ] = (uint32_t)ggml.ne[ i ];

	const ggml_type dataType = ggml.type;
	// Verify a few things
	uint32_t cbElement = (uint32_t)ggml_type_size( dataType );
	for( size_t i = 0; i < 4; i++ )
	{
		size_t stride = ggml.nb[ i ];
		if( 0 != stride % cbElement )
			return E_INVALIDARG;
		size_t nn = stride / cbElement;
		if( nn > UINT_MAX )
			return DISP_E_OVERFLOW;
		nb[ i ] = (uint32_t)nn;
	}
	return S_OK;
}

TensorShape::TensorShape( const ggml_tensor& ggml )
{
	HRESULT hr = create( ggml );
	if( FAILED( hr ) )
		throw hr;
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