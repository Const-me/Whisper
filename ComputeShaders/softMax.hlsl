// Dispatch [ nr, 1, 1 ] thread groups of this shader
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 elements: packoffset( c0 );
	uint4 strides: packoffset( c1 );
	uint nr: packoffset( c2.x );
	float inputScale: packoffset( c2.y );
}

#ifndef THREADS
static const uint THREADS = 32;
#endif

groupshared float sharedAccumulators[ THREADS ];

// Compute horizontal maximum of the numbers, and broadcast to all threads of the group.
void horizontalMaxBroadcast( const uint thread, inout float ax )
{
	sharedAccumulators[ thread ] = ax;
	for( uint i = THREADS / 2; i > 0; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			ax = max( ax, sharedAccumulators[ thread + i ] );
			sharedAccumulators[ thread ] = ax;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	ax = sharedAccumulators[ 0 ];
}

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
void horizontalSum( const uint thread, inout float sum )
{
	sharedAccumulators[ thread ] = sum;
	for( uint i = THREADS / 2; i > 1; i /= 2 )
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

static const float negativeInfinity = asfloat( 0xff800000 );

[numthreads( THREADS, 1, 1 )]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint p = group.x * strides[ 1 ];
	const uint nc = elements[ 0 ];
	const uint pEnd = p + nc;
	uint i;

	float m = negativeInfinity;
	for( i = p + thread; i < pEnd; i += THREADS )
		m = max( m, result[ i ] );
	horizontalMaxBroadcast( thread, m );

	float sum = 0;
	for( i = p + thread; i < pEnd; i += THREADS )
	{
		float f = result[ i ];

		[branch]
		if( f != negativeInfinity )
		{
			f = ( f - m ) * inputScale;
			// On both Radeon Graphics and nVidia 1080Ti, computing the exponent is slightly faster than loading from the lookup table
			f = exp( f );
			sum += f;
		}
		else
			f = 0;

		result[ i ] = f;
	}

	horizontalSum( thread, sum );
	if( 0 == thread )
		sharedAccumulators[ 0 ] = 1.0 / sum;
	GroupMemoryBarrierWithGroupSync();
	const float scale = sharedAccumulators[ 0 ];

	// ggml_vec_scale_f32
	for( i = p + thread; i < pEnd; i += THREADS )
	{
		float f = result[ i ];
		f *= scale;
		result[ i ] = f;
	}
}