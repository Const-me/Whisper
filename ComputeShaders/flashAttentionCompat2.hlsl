// Dispatch with [ ( neq1*neq2*neq3 + 31 ) / 32, 1, 1 ] thread groups
#include "flashAttentionCommon.hlsli"
Buffer<uint> lookupTable: register( t3 );

void scaleTempVector( uint i, const uint length, const float multiplier )
{
	const uint end = i + length;
	for( ; i < end; i++ )
	{
		float f = temp[ i ];
		f *= multiplier;
		// Rounding in this shader causes numerical errors on my GeForce 1080 Ti GPU, driver 527.56
		// f = roundToFp16( f );
		temp[ i ] = f;
	}
}

inline float computeTempVectorMax( uint i, const uint length )
{
	// Compute per-thread maximum
	const uint end = i + length;
	float ax = negativeInfinity;
	for( ; i < end; i++ )
		ax = max( ax, temp[ i ] );
	return ax;
}

#include "miscUtils.hlsli"
#include "fp64Utils.hlsli"

// Transform temp[ i ] = exp( temp[ i ] - tempMax ), and return the sum of these values
inline double applySoftMax( uint i, const uint length, const float tempMax )
{
	// Transform the values, and compute per-thread sum
	const uint end = i + length;
	double sum = 0;
	for( ; i < end; i++ )
	{
		float f = temp[ i ];
		[branch]
		if( f != negativeInfinity )
		{
			f -= tempMax;
			const uint index = fp16Rounded( f );
			const uint res16 = lookupTable[ index ];
			f = f16tof32( res16 );
			sum += f;
		}
		else
			f = 0;

		temp[ i ] = f;
	}
	return sum;
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 dtid: SV_DispatchThreadID )
{
	const uint neq0 = q_elements[ 0 ];
	const uint neq1 = q_elements[ 1 ];
	const uint neq2 = q_elements[ 2 ];
	const uint neq3 = q_elements[ 3 ];

	const uint nek0 = k_elements[ 0 ];
	const uint nek1 = k_elements[ 1 ];

	const uint nev1 = v_elements[ 1 ];

	const uint ne0 = res_elements[ 0 ];
	const uint ne1 = res_elements[ 1 ];

	const uint nbk0 = k_strides[ 0 ];
	const uint nbk1 = k_strides[ 1 ];
	const uint nbk2 = k_strides[ 2 ];
	const uint nbk3 = k_strides[ 3 ];

	const uint nbq0 = q_strides[ 0 ];
	const uint nbq1 = q_strides[ 1 ];
	const uint nbq2 = q_strides[ 2 ];
	const uint nbq3 = q_strides[ 3 ];

	const uint nbv0 = v_strides[ 0 ];
	const uint nbv1 = v_strides[ 1 ];
	const uint nbv2 = v_strides[ 2 ];
	const uint nbv3 = v_strides[ 3 ];

	const uint nb0 = res_strides[ 0 ];
	const uint nb1 = res_strides[ 1 ];
	const uint nb2 = res_strides[ 2 ];
	const uint nb3 = res_strides[ 3 ];

	const uint D = neq0;
	const uint N = neq1;
	const uint P = nek1 - N;
	// const uint M = P + N;
	const uint M = nek1;

	const uint ir = dtid.x;
	if( ir >= neq1 * neq2 * neq3 )
		return;

	const uint iq3 = ir / ( neq2 * neq1 );
	const uint iq2 = ( ir - iq3 * neq2 * neq1 ) / neq1;
	const uint iq1 = ( ir - iq3 * neq2 * neq1 - iq2 * neq1 );

	const uint tempIndex = ir * tempBufferStride;

	// Softmax
	float tvm = computeTempVectorMax( tempIndex, M );
	double sum = applySoftMax( tempIndex, M, tvm );

	scaleTempVector( tempIndex, M, (float)( 1.0 / sum ) );
}