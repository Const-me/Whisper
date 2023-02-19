RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint elements: packoffset( c0.x );
	bool writeNan: packoffset( c0.y );
}

// Thread group index is 16 bits per coordinate:
// https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11devicecontext-dispatch
// We want this shader to support buffers up to 2 GB.
#ifndef THREADS
static const uint THREADS = 512;
#endif
#ifndef ITERATIONS
static const uint ITERATIONS = 128;
#endif

static const uint itemsPerGroup = THREADS * ITERATIONS;

[numthreads( THREADS, 1, 1 )]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint rdi = group.x * itemsPerGroup;
	const uint rdiEnd = min( rdi + itemsPerGroup, elements );
	// https://www.h-schmidt.net/FloatConverter/IEEE754.html
	const float pattern = writeNan ? asfloat( 0x7FFFFFFFu ) : 0.0;
	for( rdi += thread; rdi < rdiEnd; rdi += THREADS )
		result[ rdi ] = pattern;
}