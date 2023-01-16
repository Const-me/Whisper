#ifndef THREADS
#define THREADS 256
#endif

// dec.tokenEmbedding tensor
Buffer<float> tokenEmbedding: register( t0 );
// dec.positionalEmbedding tensor
Buffer<float> positionalEmbedding: register( t1 );
// R32_UINT buffer with the input tokens
Buffer<uint> embd: register( t2 );
// Output tensor
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint rowLength: packoffset( c0.x );
	uint pastTokensCount: packoffset( c0.y );
	uint outputRowStride: packoffset( c0.z );
	uint2 embStrides: packoffset( c1.x );
	uint2 posStrides: packoffset( c1.z );
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint row = group.x;
	const uint rowTok = embd[ row ];
	const uint rowPos = row + pastTokensCount;

	uint rdi = row * outputRowStride;
	const uint rdiEnd = rdi + rowLength;
	rdi += thread;

	uint rsiTok = rowTok * embStrides.y;
	rsiTok += thread * embStrides.x;

	uint rsiPos = rowPos * posStrides.y;
	rsiPos += thread * posStrides.x;

	for( ; rdi < rdiEnd; rdi += THREADS, rsiTok += THREADS * embStrides.x, rsiPos += THREADS * posStrides.x )
	{
		float a = tokenEmbedding[ rsiTok ];
		float b = positionalEmbedding[ rsiPos ];
		result[ rdi ] = a + b;
	}
}