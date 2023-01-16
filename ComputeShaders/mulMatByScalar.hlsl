// Matrix * scalar product, like [ 1, E1, E2, E3 ] * [ 1, 1, E2, E3 ] = [ E1, 1, E2, E3 ]
// Dispatch [ E2, E3, 1 ] thread groups of this shader
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

inline uint hadd( uint2 vec )
{
	return vec.x + vec.y;
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const float scalarValue = arg1[ hadd( group.xy * arg1Strides.zw ) ];

	uint s0 = hadd( group.xy * arg0Strides.zw );
	const uint s0Inc = 32 * arg0Strides.y;
	s0 += thread * arg0Strides.y;

	uint rdi = hadd( group.xy * resultStrides.zw );
	const uint rdiEnd = rdi + arg0Size.y;
	rdi += thread;

	for( ; rdi < rdiEnd; rdi += 32, s0 += s0Inc )
	{
		float f = arg0[ s0 ];
		f *= scalarValue;
		result[ rdi ] = f;
	}
}