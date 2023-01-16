// Ported from ggml_compute_forward_norm_f32
// Dispatch [ ( ne01 + 31 ) / 32, ne02, ne03 ] thread groups of this shader
Buffer<float> arg0: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	uint4 result_strides: packoffset( c3 );
}

static const double eps = 1e-5; // TODO: make this a parameter

#include "groupReduce.hlsli"

double computeVectorSum( uint i, const uint length )
{
	double res = 0.0;
	const uint iEnd = i + length;
	for( ; i < iEnd; i++ )
		res += arg0[ i ];
	return res;
}

double offsetAndComputeSumSquares( uint rsi, uint rdi, const double mean, const uint length )
{
	precise double sum2 = 0.0;
	const uint rsiEnd = rsi + length;
	for( ; rsi < rsiEnd; rsi++, rdi++ )
	{
		double v = arg0[ rsi ];
		v -= mean;
		result[ rdi ] = (float)v;
		double prod = v * v;
		sum2 += prod;
	}
	return sum2;
}

void scaleVector( uint rdi, const float scale, const uint length )
{
	const uint rdiEnd = rdi + length;
	for( ; rdi < rdiEnd; rdi++ )
	{
		float f = result[ rdi ];
		f *= scale;
		result[ rdi ] = f;
	}
}

#include "fp64Utils.hlsli"

[ numthreads( 32, 1, 1 ) ]
void main( uint3 dtid: SV_DispatchThreadID )
{
	const uint i03 = dtid.z;
	const uint i02 = dtid.y;
	const uint i01 = dtid.x;
	if( i01 >= src0_elements[ 1 ] )
		return;

	const uint nb01 = src0_strides[ 1 ];
	const uint nb02 = src0_strides[ 2 ];
	const uint nb03 = src0_strides[ 3 ];

	const uint p = i01 * nb01 + i02 * nb02 + i03 * nb03;
	const uint ne00 = src0_elements[ 0 ];

	double mean = computeVectorSum( p, ne00 );
	mean = div64( mean, (double)(int)ne00 );

	const uint nb1 = result_strides[ 1 ];
	const uint nb2 = result_strides[ 2 ];
	const uint nb3 = result_strides[ 3 ];
	const uint y = i01 * nb1 + i02 * nb2 + i03 * nb3;

	const double sum2 = offsetAndComputeSumSquares( p, y, mean, ne00 );
	const float scale = (float)div64( 1.0, sqrt64( sum2 / (float)(int)ne00 + eps ) );

	scaleVector( y, scale, ne00 );
}