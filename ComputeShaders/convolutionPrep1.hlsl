// ggml_compute_forward_conv_1d_1s_f16_f32, prepare kernel data (src0)
// Dispatch [ ne01, ne02, 1 ] thread groups
Buffer<float> arg0: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
}

inline uint roundUp32( uint x )
{
	return ( x + 31 ) & ( ~31u );
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint nb01 = src0_strides[ 1 ];
	const uint nb02 = src0_strides[ 2 ];

	const uint ne00 = src0_elements[ 0 ];
	const uint ne01 = src0_elements[ 1 ];
	const uint ew0 = roundUp32( ne01 );

	const uint i02 = group.y;
	const uint i01 = group.x;

	uint rsi = i02 * nb02 + i01 * nb01;
	const uint rsiEnd = rsi + ne00;
	uint rdi = i02 * ew0 * ne00 + i01;
	rsi += thread;
	rdi += thread * ew0;
	const uint rdiInc = 32 * ew0;

	for( ; rsi < rsiEnd; rsi += 32, rdi += rdiInc )
		result[ rdi ] = arg0[ rsi ];
}