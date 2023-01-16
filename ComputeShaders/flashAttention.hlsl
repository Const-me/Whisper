// Ported from ggml_compute_forward_flash_attn_f16
// Dispatch with [ neq1*neq2*neq3, 1, 1 ] thread groups

#include "flashAttentionCommon.hlsli"
Buffer<uint> lookupTable: register( t3 );
#include "groupReduce.hlsli"

inline void computeDotProduct( Buffer<float> buff0, Buffer<float> buff1, uint s0, uint s1, const uint len, const uint thread, inout float acc )
{
	acc = 0;
	const uint s0End = s0 + len;
	s0 += thread;
	s1 += thread;
	for( ; s0 < s0End; s0 += 32, s1 += 32 )
		acc = mad( buff0[ s0 ], buff1[ s1 ], acc );

	horizontalSum( thread, acc );
}

inline void computeDotProduct( Buffer<float> buff0, RWBuffer<float> buff1, uint s0, uint s1, const uint len, const uint thread, inout float acc )
{
	acc = 0;
	const uint s0End = s0 + len;
	s0 += thread;
	s1 += thread;
	for( ; s0 < s0End; s0 += 32, s1 += 32 )
		acc = mad( buff0[ s0 ], buff1[ s1 ], acc );

	horizontalSum( thread, acc );
}

void scaleTempVector( uint i, const uint length, const uint thread, const float multiplier, bool round )
{
	const uint end = i + length;
	for( i += thread; i < end; i += 32 )
	{
		float f = temp[ i ];
		f *= multiplier;
		if( round )
			f = roundToFp16( f );
		temp[ i ] = f;
	}
}

#include "miscUtils.hlsli"

// Transform temp[ i ] = exp( temp[ i ] - tempMax ), and return the sum of these values
inline float applySoftMax( uint i, const uint length, const uint thread, const float tempMax )
{
	// Transform the values, and compute per-thread sum
	const uint end = i + length;
	float sum = 0;
	for( i += thread; i < end; i += 32 )
	{
		float f = temp[ i ];
		[branch]
		if( f != negativeInfinity )
		{
			f -= tempMax;
			const uint index = fp16Rounded( f );
			const uint res16 = lookupTable[ index ];
			f = f16tof32( res16 );
		}
		else
			f = 0;

		temp[ i ] = f;
		sum += f;
	}

	// Reduce per-thread sum to the global one, over all threads of the group
	horizontalSumBroadcast( thread, sum );
	return sum;
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
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
	const uint M = nek1;

	const uint ir = group.x;
	const uint iq3 = ir / ( neq2 * neq1 );
	const uint iq2 = ( ir - iq3 * neq2 * neq1 ) / neq1;
	const uint iq1 = ( ir - iq3 * neq2 * neq1 - iq2 * neq1 );

	const uint tempIndex = ir * tempBufferStride;

	uint ic;
	float tvm = negativeInfinity;
	const uint s1 = iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3;
	uint s0 = iq2 * nbk2 + iq3 * nbk3;
	for( ic = 0; ic < nek1; ic++, s0 += nbk1 )
	{
		if( masked )
		{
			if( ic > P + iq1 )
			{
				if( 0 == thread )
					temp[ tempIndex + ic ] = negativeInfinity;
				continue;
			}
		}

		float dp;
		computeDotProduct( k, q, s0, s1, neq0, thread, dp );
		if( 0 == thread )
		{
			dp *= scale;
			temp[ tempIndex + ic ] = dp;
			tvm = max( tvm, dp );
		}
	}

	if( 0 == thread )
		sharedAccumulators[ 0 ] = tvm;
	GroupMemoryBarrierWithGroupSync();
	tvm = sharedAccumulators[ 0 ];

	// Softmax
	{
		float sum = applySoftMax( tempIndex, M, thread, tvm );
		scaleTempVector( tempIndex, M, thread, 1.0 / sum, true );
	}

	s0 = iq2 * nbv2 + iq3 * nbv3;
	uint rdi = iq1 * nb1 + iq2 * nb2 + iq3 * nb3;
	for( ic = 0; ic < nev1; ic++, s0 += nbv1, rdi += nb0 )
	{
		float dp;
		computeDotProduct( v, temp, s0, tempIndex, nek1, thread, dp );
		if( 0 == thread )
			result[ rdi ] = dp;
	}
}