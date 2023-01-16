// ggml_compute_forward_conv_1d_1s_f16_f32, prepare source data (src1)
// Dispatch [ ne11, 1, 1 ] thread groups
Buffer<float> arg1: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src1_elements: packoffset( c2 );
	uint4 src1_strides: packoffset( c3 );
}

#include "miscUtils.hlsli"

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint i11 = group.x;

	const uint ne00 = src0_elements[ 0 ];
	const uint ne01 = src0_elements[ 1 ];
	const uint ne10 = src1_elements[ 0 ];
	const uint nb11 = src1_strides[ 1 ];

	const uint nk = ne00;
	const uint nh = nk / 2;
	const int ew0 = roundUp32( ne01 );

	uint rsi = i11 * nb11;
	uint rdi = nh * ew0 + i11;
	const uint rdiInc = ew0 * 32;
	const uint rsiEnd = rsi + ne10;

	rsi += thread;
	rdi += thread * ew0;

	for( ; rsi < rsiEnd; rsi += 32, rdi += rdiInc )
	{
		float f = arg1[ rsi ];
		f = adjustFp16( f );
		result[ rdi ] = f;
	}
}