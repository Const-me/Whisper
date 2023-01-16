// Matrix * row product, like [ E0, E1, E2, E3 ] * [ E0, 1, E2, E3 ] = [ E1, 1, E2, E3 ]
// Dispatch [ E1, E2, E3 ] groups of this shader
Buffer<float> arg0: register( t0 );
Buffer<float> arg1: register( t1 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 arg0Size: packoffset( c0 );
	uint4 arg0Strides: packoffset( c1 );
	uint4 arg1Size: packoffset( c2 );
	uint4 arg1Strides: packoffset( c3 );
	uint4 resultSize: packoffset( c4 );
	uint4 resultStrides: packoffset( c5 );
}

#include "groupReduce.hlsli"

inline uint hadd( uint3 vec )
{
	return vec.x + vec.y + vec.z;
}
inline uint hadd( uint2 vec )
{
	return vec.x + vec.y;
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint s0 = hadd( group * arg0Strides.yzw );
	uint s1 = hadd( group.yz * arg1Strides.zw );
	const uint s0End = s0 + arg0Size.x * arg0Strides.x;
	const uint s0Inc = 32 * arg0Strides.x;
	const uint s1Inc = 32 * arg1Strides.x;

	s0 += thread * arg0Strides.x;
	s1 += thread * arg1Strides.x;
	float dp = 0;
	for( ; s0 < s0End; s0 += s0Inc, s1 += s1Inc )
		dp = mad( arg0[ s0 ], arg1[ s1 ], dp );

	horizontalSum( thread, dp );
	if( 0 != thread )
		return;

	const uint rdi = group.x + hadd( group.yz * resultStrides.zw );
	result[ rdi ] = dp;
}