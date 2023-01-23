#include "stdafx.h"
#include "MlContext.h"
#include "testUtils.h"
using namespace DirectCompute;

Tensor MlContext::createTensor( eDataType type, const std::array<uint32_t, 4>& ne )
{
	Tensor res;
	check( res.create( type, ne ) );
	return res;
}

Tensor MlContext::createTensor( eDataType type, std::initializer_list<uint32_t> ne )
{
	size_t nDims = ne.size();
	if( 0 == nDims || nDims > 4 )
		throw E_INVALIDARG;
	std::array<uint32_t, 4> arr;
	for( size_t i = 0; i < nDims; i++ )
		arr[ i ] = ne.begin()[ i ];
	for( size_t i = nDims; i < 4; i++ )
		arr[ i ] = 1;
	return createTensor( type, arr );
}

Tensor MlContext::conv_1d_1s( const Tensor& a, const Tensor& b )
{
	assert( b.isMatrix() );
	assert( a.ne[ 1 ] == b.ne[ 1 ] );
	assert( a.ne[ 3 ] == 1 );

	Tensor res = createTensor( eDataType::FP32, { b.ne[ 0 ], a.ne[ 2 ] } );

	convolution( a, b, res );
	return res;
}

Tensor MlContext::conv_1d_2s( const Tensor& a, const Tensor& b )
{
	assert( b.isMatrix() );
	assert( a.ne[ 1 ] == b.ne[ 1 ] );
	assert( a.ne[ 3 ] == 1 );

	Tensor res = createTensor( eDataType::FP32, { b.ne[ 0 ] / 2, a.ne[ 2 ] } );
#if 0
	static PrintUniqueTensorSizes printSize( "conv_1d_2s" );
	printSize.print( a, b );
#endif
	convolution2( a, b, res );
	return res;
}

namespace
{
	inline bool canRepeat( const TensorShape& t0, const TensorShape& t1 )
	{
		return ( t1.ne[ 0 ] % t0.ne[ 0 ] == 0 ) &&
			( t1.ne[ 1 ] % t0.ne[ 1 ] == 0 ) &&
			( t1.ne[ 2 ] % t0.ne[ 2 ] == 0 ) &&
			( t1.ne[ 3 ] % t0.ne[ 3 ] == 0 );
	}
}

Tensor MlContext::cwiseBinary( const Tensor& a, const Tensor& b, eComputeShader cs )
{
	assert( isSameShape( a, b ) );
	Tensor res = createTensor( a.getType(), a.ne );
	cwiseBinary( a, b, res, cs );
	return res;
}

Tensor __declspec( noinline ) MlContext::view2d( const Tensor& a, uint32_t ne0, uint32_t ne1, uint32_t nb1, uint32_t offset )
{
	if( 0 != offset )
		throw E_NOTIMPL;

	Tensor res = a;
	res.ne = { ne0, ne1, 1, 1 };

	res.nb[ 1 ] = nb1;
	res.nb[ 2 ] = res.nb[ 3 ] = nb1 * ne1;
	return res;
}

Tensor MlContext::transpose( const Tensor& a )
{
	Tensor result;

	// A magic number for _mm_shuffle_epi32 SSE2 instruction to swap two lower int32 lanes in a vector
	constexpr int swapXy = _MM_SHUFFLE( 3, 2, 0, 1 );

	__m128i v = a.sizeVec();
	v = _mm_shuffle_epi32( v, swapXy );
	store( result.ne, v );

	v = a.stridesVec();
	v = _mm_shuffle_epi32( v, swapXy );
	store( result.nb, v );

	result.setGpuViews( a, a );
	return result;
}

Tensor MlContext::norm( const Tensor& a )
{
	Tensor res = createTensor( a.getType(), a.ne );
	norm( a, res );
	return res;
}

Tensor MlContext::mulMat( const Tensor& a, const Tensor& b )
{
	if( !canMulMat( a, b ) )
		throw E_INVALIDARG;
	Tensor res = createTensor( eDataType::FP32, { a.ne[ 1 ], b.ne[ 1 ], a.ne[ 2 ], b.ne[ 3 ] } );
	if constexpr( enableInexactOptimizations )
		mulMatTiled( a, b, res );
	else
		mulMat( a, b, res );
#if 0
	Tensor testTiled;
	check( testTiled.create( eDataType::FP32, res.ne ) );
	mulMatTiled( a, b, testTiled );

	std::vector<float> current, tiled;
	res.download( current );
	testTiled.download( tiled );
	sTensorDiff diff = computeDiff( current.data(), tiled.data(), current.size() );
	diff.print( "mulMatTiled" );
#endif
	return res;
}

Tensor MlContext::mulMatEx( const Tensor& a, const Tensor& b, const char* tagName )
{
	if( !canMulMat( a, b ) )
		throw E_INVALIDARG;
	if( 0 != a.nb[ 0 ] )
		throw E_INVALIDARG;	// The first argument is expected to be pre-transposed
	
	const uint16_t tag = profiler.setNextTag( tagName );

	if( b.ne[ 1 ] != 1 )
	{
		if( b.nb[ 0 ] != 0 )
		{
			Tensor rhs = reshapePanels( b );
			profiler.setNextTag( tag );
			return mulMatTiledEx( a, rhs );
		}
		else
		{
			// Second argument already reshaped into these panels
			return mulMatTiledEx( a, b );
		}
	}
	else
	{
		if( 0 != b.nb[ 0 ] )
			return mulMatByRowTiledEx( a, b );

		// That shader requires classic VRAM layout of the second argument, gonna fail with pre-transposed one
		throw E_INVALIDARG;
	}
}

Tensor MlContext::permute( const Tensor& a, uint8_t axis0, uint8_t axis1, uint8_t axis2, uint8_t axis3 )
{
	assert( axis0 < 4 );
	assert( axis1 < 4 );
	assert( axis2 < 4 );
	assert( axis3 < 4 );

	assert( axis0 != axis1 );
	assert( axis0 != axis2 );
	assert( axis0 != axis3 );
	assert( axis1 != axis2 );
	assert( axis1 != axis3 );
	assert( axis2 != axis3 );

	Tensor res = a;
	res.ne[ axis0 ] = a.ne[ 0 ];
	res.ne[ axis1 ] = a.ne[ 1 ];
	res.ne[ axis2 ] = a.ne[ 2 ];
	res.ne[ axis3 ] = a.ne[ 3 ];

	res.nb[ axis0 ] = a.nb[ 0 ];
	res.nb[ axis1 ] = a.nb[ 1 ];
	res.nb[ axis2 ] = a.nb[ 2 ];
	res.nb[ axis3 ] = a.nb[ 3 ];
	return res;
}

Tensor MlContext::flashAttention( const Tensor& q, const Tensor& k, const Tensor& v, bool masked )
{
	if( !canMulMat( k, q ) )
		throw E_INVALIDARG;

	if constexpr( enableInexactOptimizations )
	{
		if( !masked )
		{
			profiler.setNextTag( "flashAttn.1" );
			Tensor tmp = mulMat( k, q );

			profiler.setNextTag( "flashAttention" );
			const float tempScale = (float)( 1.0 / sqrt( (double)(int)q.ne[ 0 ] ) );
			softMax( tmp, tempScale );

			profiler.setNextTag( "flashAttn.2" );
			return mulMat( v, tmp );
		}
	}

	Tensor res = createTensor( eDataType::FP32, q.ne );
	flashAttention( q, k, v, res, masked );

#if 0
	Tensor tmpMat = mulMat( k, q );
	float scale = (float)( 1.0 / sqrt( (double)(int)q.ne[ 0 ] ) );
	softMax( tmpMat, scale );
	Tensor testRes = mulMat( v, tmpMat );
	computeDiff( res, testRes ).print( "flashAttention mulmat" );
#endif

	return res;
}

Tensor MlContext::copy( const Tensor& a, eDataType type, std::initializer_list<uint32_t> size )
{
	const size_t dims = size.size();
	if( 0 == dims || dims > 4 )
		throw E_BOUNDS;

	size_t nRequested = 1;
	for( size_t i = 0; i < dims; i++ )
	{
		uint32_t n = size.begin()[ i ];
		nRequested *= n;
	}
	if( nRequested != a.countElements() )
		throw E_INVALIDARG;

	const eDataType st = a.getType();
	Tensor res;
	if( a.isContinuous() && st == type )
	{
		// Same type, and it's dense - no need to call any compute shaders, equal to reshape
		res = a;
		for( size_t i = 0; i < dims; i++ )
			res.ne[ i ] = size.begin()[ i ];;
		for( size_t i = dims; i < 4; i++ )
			res.ne[ i ] = 1;
		res.setDenseStrides();
	}
	else
	{
		// Either converting non-continuous to continuous, or converting types
		res = createTensor( type, size );
		copyImpl( a, res, st == eDataType::FP32 && type == eDataType::FP16 );
	}
	return res;
}

void MlContext::copyInPlace( Tensor& dest, const Tensor& a, eDataType type, std::initializer_list<uint32_t> size )
{
	assert( type == dest.getType() );

	const size_t dims = size.size();
	if( 0 == dims || dims > 4 )
		throw E_BOUNDS;

	size_t nRequested = 1;
	for( size_t i = 0; i < dims; i++ )
	{
		uint32_t n = size.begin()[ i ];
		nRequested *= n;
	}
	if( nRequested != a.countElements() || nRequested != dest.countElements() )
		throw E_INVALIDARG;

	// Reshape the destination
	for( size_t i = 0; i < dims; i++ )
		dest.ne[ i ] = size.begin()[ i ];
	for( size_t i = dims; i < 4; i++ )
		dest.ne[ i ] = 1;
	dest.setDenseStrides();

	// Call the shader
	const eDataType st = a.getType();
	copyImpl( a, dest, st == eDataType::FP32 && type == eDataType::FP16 );
}