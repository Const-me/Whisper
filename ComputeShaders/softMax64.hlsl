// Dispatch [ nr, 1, 1 ] thread groups of this shader
RWBuffer<float> result: register( u0 );

// table_exp_f16
Buffer<uint> lookupTable: register( t0 );

cbuffer Constants: register( b0 )
{
	uint4 elements: packoffset( c0 );
	uint4 strides: packoffset( c1 );
	uint nr: packoffset( c2.x );
	float inputScale: packoffset( c2.y );
}

#include "miscUtils.hlsli"
#include "groupReduce64.hlsli"

static const float negativeInfinity = asfloat( 0xff800000 );

[ numthreads( 64, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint p = group.x * strides[ 1 ];
	const uint nc = elements[ 0 ];
	const uint pEnd = p + nc;
	uint i;

	float m = negativeInfinity;
	for( i = p + thread; i < pEnd; i += 64 )
		m = max( m, result[ i ] );
	horizontalMaxBroadcast( thread, m );

	float sum = 0;
	for( i = p + thread; i < pEnd; i += 64 )
	{
		float f = result[ i ];

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

		result[ i ] = f;
	}

	horizontalSum( thread, sum );
	if( 0 == thread )
		sharedAccumulators[ 0 ] = 1.0 / sum;
	GroupMemoryBarrierWithGroupSync();
	const float scale = sharedAccumulators[ 0 ];

	// ggml_vec_scale_f32
	for( i = p + thread; i < pEnd; i += 64 )
	{
		float f = result[ i ];
		f *= scale;
		result[ i ] = f;
	}
}