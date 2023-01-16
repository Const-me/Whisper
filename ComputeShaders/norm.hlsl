// Ported from ggml_compute_forward_norm_f32
// Dispatch [ ne01, ne02, ne03 ] thread groups of this shader
Buffer<float> arg0: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	uint4 result_strides: packoffset( c3 );
}

static const float eps = 1e-5f; // TODO: make this a parameter

#include "groupReduce.hlsli"

float computeVectorSum( uint i, const uint length, const uint thread )
{
	float res = 0.0;

	const uint iEnd = i + length;
	i += thread;
	for( ; i < iEnd; i += 32 )
		res += arg0[ i ];

	horizontalSumBroadcast( thread, res );
	return res;
}

float offsetAndComputeSumSquares( uint rsi, uint rdi, const float mean, const uint length, const uint thread )
{
	float sum2 = 0.0;

	const uint rsiEnd = rsi + length;
	rsi += thread;
	rdi += thread;
	for( ; rsi < rsiEnd; rsi += 32, rdi += 32 )
	{
		float v = arg0[ rsi ] - mean;
		result[ rdi ] = v;
		sum2 = mad( v, v, sum2 );
	}

	horizontalSumBroadcast( thread, sum2 );
	return sum2;
}

void scaleVector( uint rdi, const float scale, const uint length, const uint thread )
{
	const uint rdiEnd = rdi + length;
	for( rdi += thread; rdi < rdiEnd; rdi += 32 )
	{
		float f = result[ rdi ];
		f *= scale;
		result[ rdi ] = f;
	}
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint i03 = group.z;
	const uint i02 = group.y;
	const uint i01 = group.x;

	const uint nb01 = src0_strides[ 1 ];
	const uint nb02 = src0_strides[ 2 ];
	const uint nb03 = src0_strides[ 3 ];

	const uint p = i01 * nb01 + i02 * nb02 + i03 * nb03;

	const uint ne00 = src0_elements[ 0 ];

	float mean = computeVectorSum( p, ne00, thread );
	mean /= (float)(int)ne00;

	const uint nb1 = result_strides[ 1 ];
	const uint nb2 = result_strides[ 2 ];
	const uint nb3 = result_strides[ 3 ];
	const uint y = i01 * nb1 + i02 * nb2 + i03 * nb3;

	float sum2 = offsetAndComputeSumSquares( p, y, mean, ne00, thread );
	const float scale = 1.0 / sqrt( sum2 / (float)(int)ne00 + eps );

	scaleVector( y, scale, ne00, thread );
}