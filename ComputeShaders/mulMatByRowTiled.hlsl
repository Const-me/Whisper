// Matrix * row product, like [ E0, E1, E2, E3 ] * [ E0, 1, E2, E3 ] = [ E1, 1, E2, E3 ]
// Dispatch [ ( E1 + TILE_Y - 1 ) / TILE_Y, E2, E3 ] thread groups of this shader
// This one here is the second most expensive shader in the model, after matrix*matrix product.
// Optimized heavily, as a result the readability ain't great.

#ifndef TILE_Y
static const uint TILE_Y = 64;
#endif
#ifndef THREADS_X
static const uint THREADS_X = 32;
#endif
#ifndef THREADS_Y
static const uint THREADS_Y = 16;
#endif

Buffer<float> arg0: register( t0 );
Buffer<float> arg1: register( t1 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 arg0Size: packoffset( c0 );
	uint4 arg0Strides: packoffset( c1 );
	uint4 arg1Size: packoffset( c2 );
	uint4 arg1Strides: packoffset( c3 );
	uint4 resultSize: packoffset( c4 );
	uint4 resultStrides: packoffset( c5 );
}

inline uint hadd( uint2 vec )
{
	return vec.x + vec.y;
}

// Count of FP32 accumulators we need in every thread of the shader
static const uint heightScalars = TILE_Y / THREADS_Y;
// The local accumulators are float4 vectors, compute count of these vectors
static const uint heightVectors = ( heightScalars + 3 ) / 4;

groupshared float4 reductionBuffer[ heightVectors ][ THREADS_Y ][ THREADS_X ];

[numthreads( THREADS_X, THREADS_Y, 1 )]
void main( uint3 group: SV_GroupID, uint3 thread : SV_GroupThreadID )
{
	uint i;
	// Despite inside GPU cores, the shared memory is still much slower than registers
	// For this reason, this shader accumulates numbers in local variables. Only uses groupshared buffer for the final reduction.
	float4 acc[ heightVectors ];
	// Zero out the accumulators
	[unroll]
	for( i = 0; i < heightVectors; i++ )
		acc[ i ] = 0.0;

	// Count of rows to compute in this thread group
	const uint height = min( TILE_Y, arg0Size.y - group.x * TILE_Y );

	uint s0 = hadd( group.yz * arg0Strides.zw );   //< arg0 layer for the thread group
	s0 += group.x * TILE_Y * arg0Strides.y;        //< arg0 first row for the thread group
	s0 += hadd( arg0Strides.xy * thread.xy );      //< arg0 load index for the thread

	uint s1 = hadd( group.yz * arg1Strides.zw );   //< arg1 layer for the thread group
	s1 += thread.x * arg1Strides.x;                //< arg1 load index for the thread

	const uint completeTiles = arg0Size.x / THREADS_X;
	// Each iteration of that loop loads THREADS_X elements from arg1,
	// a block of [ THREADS_X, height ] elements from arg0,
	// and accumulates these dot products in the local variables
	for( uint t = 0; t < completeTiles; t++, s0 += THREADS_X * arg0Strides.x, s1 += THREADS_X * arg1Strides.x )
	{
		// Load THREADS_X elements from arg1
		const float v1 = arg1[ s1 ];

		uint rsi = s0;
		[unroll]
		for( i = 0; i < heightVectors; i++ )
		{
			float4 v0 = 0.0;
			// Load up to 4*THREADS_X elements from arg0
			[unroll]
			for( uint j = 0; j < 4; j++, rsi += arg0Strides.y * THREADS_Y )
			{
				const uint y = ( i * 4 + j ) * THREADS_Y + thread.y;
				[branch]
				if( y < height )
					v0[ j ] = arg0[ rsi ];
			}
			// Multiply + accumulate
			acc[ i ] = mad( v0, v1, acc[ i ] );
		}
	}

	const uint rem = arg0Size.x % THREADS_X;
	if( thread.x < rem )
	{
		// E0 ain't a multiple of THREADS_X, we have a remainder

		// Load `rem` elements from arg1
		const float v1 = arg1[ s1 ];

		[unroll]
		for( i = 0; i < heightVectors; i++ )
		{
			float4 v0 = 0.0;
			// Load up to 4*rem elements from arg0
			[unroll]
			for( uint j = 0; j < 4; j++, s0 += arg0Strides.y * THREADS_Y )
			{
				const uint y = ( i * 4 + j ) * THREADS_Y + thread.y;
				[branch]
				if( y < height )
					v0[ j ] = arg0[ s0 ];
			}
			// Multiply + accumulate
			acc[ i ] = mad( v0, v1, acc[ i ] );
		}
	}

	// Now we need horizontal sum of these accumulators, reducing [height][THREADS_X] of them into [height][1] column
	// First, store local variables into the shared memory.
	[ unroll ]
	for( i = 0; i < heightVectors; i++ )
		reductionBuffer[ i ][ thread.y ][ thread.x ] = acc[ i ];
	GroupMemoryBarrierWithGroupSync();

	// Run reduction using that shared memory buffer
	for( i = THREADS_X / 2; i > 1; i /= 2 )
	{
		if( thread.x < i )
		{
			[unroll]
			for( uint iv = 0; iv < heightVectors; iv++ )
			{
				float4 that = reductionBuffer[ iv ][ thread.y ][ thread.x + i ];
				float4 tmp = acc[ iv ];
				tmp += that;
				reductionBuffer[ iv ][ thread.y ][ thread.x ] = tmp;
				acc[ iv ] = tmp;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// And finally, store that column to global memory.
	// Only running that code on the threads of the group with thread.x = 0, to save a few loads from the groupshared buffer
	// This allows to use registers instead, faster to access
	if( thread.x != 0 )
		return;

	uint rdi = hadd( group.yz * resultStrides.zw );
	rdi += ( group.x * TILE_Y + thread.y ) * resultStrides.x;
	const uint rdiInc = THREADS_Y * resultStrides.x;

	[unroll]
	for( i = 0; i < heightVectors; i++ )
	{
		// The previous loop had "i > 1" continue condition, it didn't complete the last step of the reduction
		// The following line is doing that last reduction step
		const float4 resultVec = acc[ i ] + reductionBuffer[ i ][ thread.y ][ 1 ];

		// Conditionally store these 4 floats to the output tensor
		[unroll]
		for( uint j = 0; j < 4; j++, rdi += rdiInc )
		{
			const uint y = ( i * 4 + j ) * THREADS_Y + thread.y;
			[branch]
			if( y < height )
				result[ rdi ] = resultVec[ j ];
		}
	}
}