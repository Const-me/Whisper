// When reset = TRUE, write zero to the output buffer
// When reset = FALSE, test input tensor for NaN, when found at least 1 NaN element, write 1 to the output buffer

// FP32 or FP16 tensor to test for NAN
Buffer<float> tensor: register( t0 );
// A buffer with a single element for the output boolean. Zero means there were no NAN values in the tensor.
RWBuffer<uint> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint elements: packoffset( c0.x );
	bool reset : packoffset( c0.y );
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

inline bool isNaN( float x )
{
	// https://sakibsaikia.github.io/graphics/2022/01/04/Nan-Checks-In-HLSL.html
	return ( asuint( x ) & 0x7fffffff ) > 0x7f800000;
}

groupshared uint reductionBuffer;

[numthreads( THREADS, 1, 1 )]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	if( reset )
	{
		if( 0 == thread )
			result[ 0 ] = 0;
		return;
	}

	uint rsi = group.x * itemsPerGroup;
	const uint rsiEnd = min( rsi + itemsPerGroup, elements );

	// The main loop updates a local variable. There're THREADS instances of that variable for the group of threads.
	bool foundNan = false;
	for( rsi += thread; rsi < rsiEnd; rsi += THREADS )
	{
		const float val = tensor[ rsi ];
		if( !isNaN( val ) )
			continue;
		foundNan = true;
		break;
	}

	// Reduce THREADS booleans to a single one, using group shared memory atomics
	if( 0 == thread )
		reductionBuffer = 0;
	GroupMemoryBarrierWithGroupSync();

	if( foundNan )
		InterlockedOr( reductionBuffer, 1u );

	GroupMemoryBarrierWithGroupSync();

	// When found, update output value with global memory atomic
	if( 0 != thread )
		return;
	if( 0 == reductionBuffer )
		return;

	InterlockedOr( result[ 0 ], 1u );
}