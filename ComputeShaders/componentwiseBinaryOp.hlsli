Buffer<float> arg0: register( t0 );
Buffer<float> arg1: register( t1 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	uint4 src1_elements: packoffset( c2 );
	uint4 src1_strides: packoffset( c3 );
	uint4 result_elements: packoffset( c4 );
	uint4 result_strides: packoffset( c5 );
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint j = group.x;
	const uint nb1 = result_strides[ 1 ];
	const uint nb01 = src0_strides[ 1 ];

	const uint nb10 = src1_strides[ 0 ];
	const uint nb11 = src1_strides[ 1 ];
	const uint nc = src0_elements[ 0 ];

	uint rsi0 = j * nb01;
	uint rsi1 = j * nb11;
	uint rdi = j * nb1;
	const uint rsi0End = rsi0 + nc;

	rsi0 += thread;
	rsi1 += thread * nb10;
	rdi += thread;

	const uint rsi1Inc = 32 * nb10;
	for( ; rsi0 < rsi0End; rsi0 += 32, rsi1 += rsi1Inc, rdi += 32 )
	{
		const float a = arg0[ rsi0 ];
		const float b = arg1[ rsi1 ];
		const float res = compute( a, b );
		result[ rdi ] = res;
	}
}