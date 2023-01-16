// When GPUs are converting FP32 to FP16, they always truncate towards 0, documented there:
// https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-data-conversion#conververting-from-a-higher-range-representation-to-a-lower-range-representation
// Whisper code uses _mm_cvtps_ph( x, 0 ), the 0 stands for "Round to nearest even": https://www.felixcloutier.com/x86/vcvtps2ph
// This function adjusts FP32 value making it so that truncation towards 0 results in the value equal to what CPU is doing
inline float adjustFp16( const float src )
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
			return src;
		}
		else if( errTrunc > errNext )
		{
			// Truncated + 1 was closer to the source
			return next32;
		}
		else
		{
			// Exactly half, doing banker's rounding to nearest even
			return ( 0 == ( trunc16 & 1 ) ) ? src : next32;
		}
	}
	else
	{
		// INF or NAN
		return src;
	}
}

// Convert FP32 number to FP16, using rounding to nearest
inline uint fp16Rounded( const float src )
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
			return trunc16;
		}
		else if( errTrunc > errNext )
		{
			// Truncated + 1 was closer to the source
			return next16;
		}
		else
		{
			// Exactly half, doing banker's rounding to nearest even
			return ( 0 == ( trunc16 & 1 ) ) ? trunc16 : next16;
		}
	}
	else
	{
		// INF or NAN
		return trunc16;
	}
}

// Round up the number to be a multiple of 32
inline uint roundUp32( uint x )
{
	return ( x + 31 ) & ( ~31u );
}