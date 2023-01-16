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

inline uint hadd( uint3 vec )
{
	return vec.x + vec.y + vec.z;
}
inline uint hadd( uint2 vec )
{
	return vec.x + vec.y;
}

// No idea why, but that particular configuration appears to be the fastest one on Ryzen 7 5700G iGPU
// Not by much, though: when trying a few numbers I saw 1.30 - 1.42 seconds for this compute shader
static const uint THREADS = 64;
static const uint REDUCTION_BUFFER = 32;
groupshared float sharedAccumulators[ REDUCTION_BUFFER ];

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
void horizontalSum( const uint thread, inout float sum )
{
	if( THREADS > REDUCTION_BUFFER )
	{
		for( uint t = REDUCTION_BUFFER; t < THREADS; t += REDUCTION_BUFFER )
		{
			// Threads [ t .. t + REDUCTION_BUFFER ] store into the buffer
			if( thread >= t && thread < t + REDUCTION_BUFFER )
				sharedAccumulators[ thread - t ] = sum;

			GroupMemoryBarrierWithGroupSync();

			// Threads [ 0 .. REDUCTION_BUFFER ] increment their local sum with the value loaded from the buffer
			if( thread < REDUCTION_BUFFER )
				sum += sharedAccumulators[ thread ];
		}
	}

	if( thread < REDUCTION_BUFFER )
		sharedAccumulators[ thread ] = sum;

	for( uint i = REDUCTION_BUFFER / 2; i > 1; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			sum += sharedAccumulators[ thread + i ];
			sharedAccumulators[ thread ] = sum;
		}
	}

	GroupMemoryBarrierWithGroupSync();
	if( 0 == thread )
		sum += sharedAccumulators[ 1 ];
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint s0 = hadd( group * arg0Strides.yzw );
	uint s1 = hadd( group.yz * arg1Strides.zw );
	const uint s0End = s0 + arg0Size.x * arg0Strides.x;
	const uint s0Inc = THREADS * arg0Strides.x;
	const uint s1Inc = THREADS * arg1Strides.x;

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