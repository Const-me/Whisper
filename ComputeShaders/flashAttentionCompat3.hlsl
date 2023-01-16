// Dispatch with [ neq1*neq2*neq3, 1, 1 ] thread groups
#include "flashAttentionCommon.hlsli"
#include "groupReduce.hlsli"
#include "miscUtils.hlsli"

inline void roundTempVector( uint i, const uint len, const uint thread )
{
	const uint iEnd = i + len;
	for( i += thread; i < iEnd; i += 32 )
	{
		float f = temp[ i ];
		f = roundToFp16( f );
		temp[ i ] = f;
	}
}

inline void computeDotProduct( Buffer<float> buff0, RWBuffer<float> buff1, uint s0, uint s1, const uint len, const uint thread, inout float acc )
{
	acc = 0;
/*	const uint s0End = s0 + len;
	s0 += thread;
	s1 += thread;
	for( ; s0 < s0End; s0 += 32, s1 += 32 )
		acc = mad( buff0[ s0 ], buff1[ s1 ], acc );

	horizontalSumCompatNew( thread, acc ); */
	const uint completeVectors = len / 32;
	uint i;
	for( i = 0; i < completeVectors; i++, s0 += 32, s1 += 32 )
		acc = mad( buff0[ s0 + thread ], buff1[ s1 + thread ], acc );

	horizontalSumCompatNew( thread, acc );

	if( 0 == thread )
	{
		const uint rem = len % 32;
		if( 0 != rem )
		{
			double f64 = acc;
			for( i = 0; i < rem; i++ )
			{
				precise float a = buff0[ s0 + i ];
				precise float b = buff1[ s1 + i ];
				precise float prod = a * b;
				f64 += prod;
			}
			acc = (float)f64;
		}
	}
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
	// const uint M = P + N;
	const uint M = nek1;

	const uint ir = group.x;
	const uint iq3 = ir / ( neq2 * neq1 );
	const uint iq2 = ( ir - iq3 * neq2 * neq1 ) / neq1;
	const uint iq1 = ( ir - iq3 * neq2 * neq1 - iq2 * neq1 );

	const uint tempIndex = ir * tempBufferStride;

	roundTempVector( tempIndex, nek1, thread );
	AllMemoryBarrierWithGroupSync();

	uint rdi = iq1 * nb1 + iq2 * nb2 + iq3 * nb3;
	for( uint ic = 0; ic < nev1; ic++, rdi += nb0 )
	{
		// dst indices
		const uint i1 = iq1;
		const uint i2 = iq2;
		const uint i3 = iq3;

		const uint s0 = ic * nbv1 + i2 * nbv2 + i3 * nbv3;
		float dp;
		computeDotProduct( v, temp, s0, tempIndex, nek1, thread, dp );
		if( 0 == thread )
			result[ rdi ] = dp;
	}
}