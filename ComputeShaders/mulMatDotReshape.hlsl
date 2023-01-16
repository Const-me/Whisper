// GGML_TASK_INIT step for matrix*matrix product, where nb01 >= nb00;
// Dispatch with [ ne11, ne12 ] groups
Buffer<float> arg0: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
}

#include "miscUtils.hlsli"

// Each thread group of this shader copies a single rows of the matrix
[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint i12 = group.y;
	const uint i11 = group.x;
	const uint ne10 = src0_elements.x;
	const uint ne11 = src0_elements.y;
	const uint nb12 = src0_strides.z;
	const uint nb11 = src0_strides.y;

	uint rdi = i11 * ne10 + i12 * ne10 * ne11;
	const uint rdiEnd = rdi + ne10;
	uint rsi = i12 * nb12 + i11 * nb11;
	rdi += thread;
	rsi += thread;

	for( ; rdi < rdiEnd; rdi += 32, rsi += 32 )
		result[ rdi ] = adjustFp16( arg0[ rsi ] );
}