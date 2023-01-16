// ggml_compute_forward_conv_1d_1s_f16_f32, GGML_TASK_COMPUTE implementation
// Dispatch [ ne10, ne02, 1 ] thread groups
Buffer<float> arg0: register( t0 );
Buffer<float> arg1: register( t1 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	uint4 src1_elements: packoffset( c2 );
	uint4 result_elements: packoffset( c4 );
	uint4 result_strides: packoffset( c5 );
}

#include "groupReduce.hlsli"

inline void computeDotProduct( uint s0, uint s1, uint len, uint thread, inout float acc )
{
	float curr = 0;
	const uint completeVectors = len / 32;
	uint i;
	for( i = 0; i < completeVectors; i++, s0 += 32, s1 += 32 )
		curr = mad( arg0[ s0 + thread ], arg1[ s1 + thread ], curr );

	horizontalSumCompatNew( thread, curr );

	if( 0 == thread )
	{
		const uint rem = len % 32;
		if( 0 != rem )
		{
			double f64 = curr;
			for( i = 0; i < rem; i++ )
			{
				precise float a = arg0[ s0 + i ];
				precise float b = arg1[ s1 + i ];
				precise float prod = a * b;
				f64 += prod;
			}
			curr = (float)f64;
		}
		acc += curr;
	}
}

#include "miscUtils.hlsli"

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint i1 = group.y;
	const uint i0 = group.x;

	const uint ne00 = src0_elements[ 0 ];
	const uint nk = ne00;
	const int nh = (int)( nk / 2 );

	const uint ne01 = src0_elements[ 1 ];
	const int ew0 = roundUp32( ne01 );

	float res = 0;
	for( int k = -nh; k <= nh; k++ )
	{
		const uint source0 = i1 * ew0 * ne00 + uint( nh + k ) * ew0;
		const uint source1 = uint( i0 + nh + k ) * ew0;
		computeDotProduct( source0, source1, ew0, thread, res );
	}

	if( 0 != thread )
		return;

	const uint nb1 = result_strides[ 1 ];
	const uint rdi = i1 * nb1 + i0;
	result[ rdi ] = res;
}