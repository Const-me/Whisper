// Ported from ggml_compute_forward_flash_attn_f16
// Dispatch with [ neq1*neq2*neq3, 1, 1 ] thread groups
Buffer<float> q: register( t0 );
Buffer<float> k: register( t1 );
Buffer<float> v: register( t2 );

RWBuffer<float> result: register( u0 );
// This temporary buffer should fit tempBufferStride * neq1 * neq2 * neq3 elements, FP32 precision
RWBuffer<float> temp: register( u1 );

cbuffer Constants: register( b0 )
{
	uint4 q_elements: packoffset( c0 );
	uint4 q_strides: packoffset( c1 );
	uint4 k_elements: packoffset( c2 );
	uint4 k_strides: packoffset( c3 );
	uint4 v_elements: packoffset( c4 );
	uint4 v_strides: packoffset( c5 );
	uint4 res_elements: packoffset( c6 );
	uint4 res_strides: packoffset( c7 );

	bool masked : packoffset( c8.x );
	// 1.0 / sqrt( (double) D )
	float scale : packoffset( c8.y );
	// This number is required to be >= nek1, and ideally rounded up to either 32 (L2 line) or 128 (L1 line) bytes
	uint tempBufferStride: packoffset( c8.z );
}

static const float negativeInfinity = asfloat( 0xff800000 );

// Convert FP32 number to FP16 using rounding to nearest, then upcast back to FP32
inline float roundToFp16( const float src )
{
	const uint trunc16 = f32tof16( src );
	const float trunc32 = f16tof32( trunc16 );

	const uint truncExp = ( trunc16 >> 10 ) & 0x1F;
	if( truncExp != 0x1F )
	{
		const uint next16 = trunc16 + 1;
		const float next32 = f16tof32( next16 );

		const float errTrunc = abs( src - trunc32 );
		const float errNext = abs( src - next32 );

		if( errTrunc < errNext )
		{
			// Truncated was closer to the source
			return trunc32;
		}
		else if( errTrunc > errNext )
		{
			// Truncated + 1 was closer to the source
			return next32;
		}
		else
		{
			// Exactly half, doing banker's rounding to nearest even
			return ( 0 == ( trunc16 & 1 ) ) ? trunc32 : next32;
		}
	}
	else
	{
		// INF or NAN
		return trunc32;
	}
}