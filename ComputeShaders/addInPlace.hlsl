#ifndef THREADS
#define THREADS 512
#endif

Buffer<float> arg0: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 size: packoffset( c0 );
	uint4 strides: packoffset( c1 );
	uint4 argStrides: packoffset( c3 );
}

inline uint rowOffset( uint3 idx, uint4 strides )
{
	return idx[ 0 ] * strides[ 1 ] + idx[ 1 ] * strides[ 2 ] + idx[ 2 ] * strides[ 3 ];
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = rowOffset( group, strides );
	uint rsi = rowOffset( group, argStrides );

	const uint rdiEnd = rdi + size[ 0 ] * strides[ 0 ];
	rdi += thread * strides[ 0 ];
	rsi += thread * argStrides[ 0 ];

	const uint rdiInc = THREADS * strides[ 0 ];
	const uint rsiInc = THREADS * argStrides[ 0 ];

	for( ; rdi < rdiEnd; rdi += rdiInc, rsi += rsiInc )
	{
		float f = result[ rdi ];
		f += arg0[ rsi ];
		result[ rdi ] = f;
	}
}