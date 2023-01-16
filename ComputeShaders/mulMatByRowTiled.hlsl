// Matrix * row product, like [ E0, E1, E2, E3 ] * [ E0, 1, E2, E3 ] = [ E1, 1, E2, E3 ]
// Dispatch [ ( E1 + TILE_Y - 1 ) / TILE_Y, E2, E3 ] thread groups of this shader

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

groupshared float resTemp[ TILE_Y ][ THREADS_X ];

inline uint hadd( uint2 vec )
{
	return vec.x + vec.y;
}

[ numthreads( THREADS_X, THREADS_Y, 1 ) ]
void main( uint3 group: SV_GroupID, uint3 thread : SV_GroupThreadID, uint threadFlattenned : SV_GroupIndex )
{
	uint i;
	// Zero out the shared buffer
	for( i = thread.y; i < TILE_Y; i += THREADS_Y )
		resTemp[ i ][ thread.x ] = 0.0;
	GroupMemoryBarrierWithGroupSync();

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
	// and accumulates these dot products in the shared buffer
	for( uint t = 0; t < completeTiles; t++, s0 += THREADS_X * arg0Strides.x, s1 += THREADS_X * arg1Strides.x )
	{
		// Load THREADS_X elements from arg1
		const float v1 = arg1[ s1 ];

		uint rsi = s0;
		for( i = thread.y; i < height; i += THREADS_Y, rsi += arg0Strides.y * THREADS_Y )
		{
			// Load THREADS_X elements from arg0
			const float v0 = arg0[ rsi ];
			// Multiply and accumulate in the shared buffer
			float acc = resTemp[ i ][ thread.x ];
			acc = mad( v0, v1, acc );
			resTemp[ i ][ thread.x ] = acc;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	const uint rem = arg0Size.x % THREADS_X;
	if( rem != 0 )
	{
		// E0 ain't a multiple of THREADS_X, we have a remainder
		float v1;
		if( thread.x < rem )
			v1 = arg1[ s1 ];
		else
			v1 = 0.0;

		for( i = thread.y; i < height; i += THREADS_Y, s0 += arg0Strides.y * THREADS_Y )
		{
			if( thread.x >= rem )
				continue;
			const float v0 = arg0[ s0 ];
			float acc = resTemp[ i ][ thread.x ];
			acc = mad( v0, v1, acc );
			resTemp[ i ][ thread.x ] = acc;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// Now we need horizontal sums of these shared accumulators, i.e. reduce [height][THREADS_X] shared array into [height][1] column
	for( i = THREADS_X / 2; i > 0; i /= 2 )
	{
		if( thread.x < i )
		{
			for( uint j = thread.y; j < height; j += THREADS_Y )
			{
				float sum = resTemp[ j ][ thread.x ];
				sum += resTemp[ j ][ thread.x + i ];
				resTemp[ j ][ thread.x ] = sum;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// And finally, store that column to global memory
	if( threadFlattenned >= height )
		return;

	uint rdi = hadd( group.yz * resultStrides.zw ) + group.x * TILE_Y * resultStrides.x;
	rdi += threadFlattenned * resultStrides.x;
	result[ rdi ] = resTemp[ threadFlattenned ][ 0 ];
}