// ggml_compute_forward_dup_f32 when we actually need to reshape the tensor
// Dispatch [ ne01, ne02, ne03 ] thread groups of this shader
Buffer<float> arg0: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	bool downcastFp32 : packoffset( c2.x );
}

#include "miscUtils.hlsli"

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint nb00 = src0_strides[ 0 ];
	const uint nb01 = src0_strides[ 1 ];
	const uint nb02 = src0_strides[ 2 ];
	const uint nb03 = src0_strides[ 3 ];

	const uint ne00 = src0_elements[ 0 ];
	const uint ne01 = src0_elements[ 1 ];
	const uint ne02 = src0_elements[ 2 ];
	const uint ne03 = src0_elements[ 3 ];

	const uint i01 = group.x;
	const uint i02 = group.y;
	const uint i03 = group.z;

	// We need following integer: i01*ne00 + i02*ne00*ne01 + i03*ne00*ne01*ne02
	// We want to minimize count of integer multiplications
	// Also, DXBC assembly features `imad` instruction which computes a*b+c for integers, the actual hardware hopefully has an equivalent
	// i03*ne00*ne01*ne02 + i02*ne00*ne01 + i01*ne00
	// ( i03*ne01*ne02 + i02*ne01 + i01 ) * ne00
	// ( ( i03*ne02 + i02) * ne01 + i01 ) * ne00
	uint rdi = ( ( i03 * ne02 + i02 ) * ne01 + i01 ) * ne00;

	const uint rdiEnd = rdi + ne00;

	uint rsi = i01 * nb01 + i02 * nb02 + i03 * nb03;
	const uint rsiInc = 32 * nb00;

	rdi += thread;
	rsi += thread * nb00;

	for( ; rdi < rdiEnd; rdi += 32, rsi += rsiInc )
	{
		float f = arg0[ rsi ];
		[branch]
		if( downcastFp32 )
			f = adjustFp16( f );
		result[ rdi ] = f;
	}
}