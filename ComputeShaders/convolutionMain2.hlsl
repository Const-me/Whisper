// ggml_compute_forward_conv_1d_2s_f16_f32, GGML_TASK_COMPUTE implementation
// Dispatch [ ne10 / 2, ne02, 1 ] thread groups
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
	const uint s0End = s0 + len;
	s0 += thread;
	s1 += thread;
	for( ; s0 < s0End; s0 += 32, s1 += 32 )
		curr = mad( arg0[ s0 ], arg1[ s1 ], curr );

	horizontalSumCompatNew( thread, curr );
	if( 0 == thread )
		acc += curr;
}

#include "miscUtils.hlsli"

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint ne00 = src0_elements[ 0 ];
	const uint ne01 = src0_elements[ 1 ];
	const int ew0 = roundUp32( ne01 );

	float res = 0;
	uint s0 = group.y * ew0 * ne00;
	uint s1 = group.x * 2 * ew0;
	// The original implementation did following:
	// int nh = (int)( nk / 2 );
	// for( int k = -nh; k <= nh; k++ )
	// What we doing instead:
	// for( uint len = ( nk / 2 ) * 2 + 1, i = 0; i < len; i++ )
	// len = ( nk / 2 ) * 2 + 1 is equal to ( nk | 1 )
	const uint s0End = s0 + ( ne00 | 1u ) * ew0;
	for( ; s0 < s0End; s0 += ew0, s1 += ew0 )
		computeDotProduct( s0, s1, ew0, thread, res );

	if( 0 != thread )
		return;

	const uint nb1 = result_strides[ 1 ];
	const uint rdi = group.y * nb1 + group.x;
	result[ rdi ] = res;
}