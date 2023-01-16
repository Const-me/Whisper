// Special softMax shader for matrices with rows of 1500 elements.
// Uses group shared buffer of that length to save global memory bandwidth, more than 2x faster than the original.
// Dispatch [ nr, 1, 1 ] thread groups of this shader
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 elements: packoffset( c0 );
	uint4 strides: packoffset( c1 );
	uint nr: packoffset( c2.x );
	float inputScale: packoffset( c2.y );
}

#include "miscUtils.hlsli"
#include "groupReduce64.hlsli"

static const uint THREADS = 64;
static const uint ROW_LENGTH = 1500;
groupshared float rowBuffer[ ROW_LENGTH ];

static const float negativeInfinity = asfloat( 0xff800000 );

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint p = group.x * strides[ 1 ];
	const uint nc = ROW_LENGTH;
	uint i;

	float m = negativeInfinity;
	// First pass: compute maximum, and copy the row into the group shared buffer
	for( i = thread; i < nc; i += THREADS )
	{
		float f = result[ p + i ];
		m = max( m, f );
		rowBuffer[ i ] = f;
	}
	horizontalMaxBroadcast( thread, m );

	// Second pass: apply initial scale, compute the exponent, and compute total sum over the row
	float sum = 0;
	for( i = thread; i < nc; i += THREADS )
	{
		float f = rowBuffer[ i ];

		[branch]
		if( f != negativeInfinity )
		{
			f = ( f - m ) * inputScale;
#if 1
			// At least on Radeon Graphics GPU inside Ryzen 7 5700G, computing exponent instead of loading from the buffer improves the performance
			f = exp( f );
#else
			uint s = fp16Rounded( f );
			s = lookupTable[ s ];
			f = f16tof32( s );
#endif
			sum += f;
		}
		else
			f = 0;

		rowBuffer[ i ] = f;
	}

	horizontalSum( thread, sum );
	if( 0 == thread )
		sharedAccumulators[ 0 ] = 1.0 / sum;
	GroupMemoryBarrierWithGroupSync();
	const float scale = sharedAccumulators[ 0 ];

	// Final pass: apply the final scale, and copy the row from the group shared buffer back into the global memory
	for( i = thread; i < nc; i += THREADS )
	{
		float f = rowBuffer[ i ];
		f *= scale;
		result[ p + i ] = f;
	}
}