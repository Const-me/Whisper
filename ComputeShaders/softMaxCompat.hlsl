// ggml_compute_forward_soft_max_f32
// Dispatch [ ( nr + 31 ) / 32, 1, 1 ] thread groups of this shader
RWBuffer<float> result: register( u0 );

// table_exp_f16
Buffer<uint> lookupTable: register( t0 );

cbuffer Constants: register( b0 )
{
	uint4 elements: packoffset( c0 );
	uint4 strides: packoffset( c1 );
	uint nr: packoffset( c2.x );
}

#include "miscUtils.hlsli"
#include "fp64Utils.hlsli"

static const float negativeInfinity = asfloat( 0xff800000 );

[ numthreads( 32, 1, 1 ) ]
void main( uint3 dtid: SV_DispatchThreadID )
{
	if( dtid.x >= nr )
		return;

	const uint p = dtid.x * strides[ 1 ];
	const uint nc = elements[ 0 ];
	const uint pEnd = p + nc;
	uint i;

	float m = negativeInfinity;
	for( i = p; i < pEnd; i++ )
		m = max( m, result[ i ] );

	double sum = 0;
	for( i = p; i < pEnd; i++ )
	{
		float f = result[ i ];

		[branch]
		if( f != negativeInfinity )
		{
			uint s = fp16Rounded( f - m );
			s = lookupTable[ s ];
			f = f16tof32( s );
			sum += f;
		}
		else
			f = 0;

		result[ i ] = f;
	}

	const float scale = (float)div64( 1.0, sum );
	// ggml_vec_scale_f32
	for( i = p; i < pEnd; i++ )
	{
		float f = result[ i ];
		f *= scale;
		result[ i ] = f;
	}
}