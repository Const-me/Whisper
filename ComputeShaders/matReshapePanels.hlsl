// This shader reshapes a matrix into the shape expected by mulMatTiledEx.hlsl and mulMatByRowTiledEx.hlsl compute shaders
// It's called in runtime, also while loading models from disk.
// So far, it's only used when running on AMD GPUs.
#ifndef TILE_SIZE
static const uint TILE_SIZE = 32;
#endif

// Input tensor
Buffer<float> source: register( t0 );
// Output tensor
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 arg0Size: packoffset( c0 );
	uint4 arg0Strides: packoffset( c1 );
	// Count of elements per panel
	uint panelSize : packoffset( c2.y );
	// Layer strides of the output matrix
	uint2 layerStrides: packoffset( c2.z );
}

inline uint hadd( uint2 v2 ) { return v2.x + v2.y; }

groupshared float tileBuffer[ TILE_SIZE ][ TILE_SIZE ];

[ numthreads( TILE_SIZE, 1, 1 ) ]
void main( const uint3 group: SV_GroupID, const uint thread : SV_GroupIndex )
{
	uint rdi = hadd( group.yz * layerStrides );
	rdi += group.x * panelSize;
	rdi += thread;

	uint rsi = hadd( group.yz * arg0Strides.zw );
	const uint baseY = group.x * TILE_SIZE;
	const uint dispatchThread = baseY + thread;
	// Reshaping into a column major horizontal panel, height = TILE_SIZE, width = width of the source matrix
	uint width = arg0Size.x;
	// Usually TILE_SIZE; can be less for the last panel on the matrix when we need to generate zeros instead of loading these numbers
	const uint height = min( TILE_SIZE, arg0Size.y - baseY );

	if( arg0Strides.x == 1 )
	{
		// The input matrix is row major, can improve performance with coalesced loads and group shared buffer.
		rsi += baseY * arg0Strides.y;

		const uint widthCompleteTiles = width / TILE_SIZE;

		if( height < TILE_SIZE )
		{
			// This thread group was dispatched for the last panel of the matrix, it doesn't have enough rows
			// Write zeros to the corresponding elements of the groupshared buffer
			for( uint j = height; j < TILE_SIZE; j++ )
				tileBuffer[ thread ][ j ] = 0.0;
		}

		for( uint i = 0; i < widthCompleteTiles; i++, rsi += TILE_SIZE )
		{
			// Load [ TILE_SIZE ] * [ TILE_SIZE ] block with fully coalesced loads, store to group shared buffer in transposed order
			uint rsiTile = rsi + thread;
			uint j;
			for( j = 0; j < height; j++, rsiTile += arg0Strides.y )
			{
				// Each iteration of the loop loads a row of [ TILE_SIZE ] elements from the corresponding row of the source tensor
				// Fully coalesced load
				float f = source[ rsiTile ];
				// Random store but the local memory's fast, this works rather well in practice
				tileBuffer[ thread ][ j ] = f;
			}

			GroupMemoryBarrierWithGroupSync();

			// Copy from group shared buffer to output tensor
			for( j = 0; j < TILE_SIZE; j++, rdi += TILE_SIZE )
			{
				// Fully coalesced loads and stores
				float f = tileBuffer[ j ][ thread ];
				result[ rdi ] = f;
			}

			GroupMemoryBarrierWithGroupSync();
		}

		width %= TILE_SIZE;
		if( 0 == width )
			return;
		rsi += thread * arg0Strides.y;
	}
	else
		rsi += dispatchThread * arg0Strides.y;

	for( uint i = 0; i < width; i++ )
	{
		float f;
		[branch]
		if( thread < height )
			f = source[ rsi ];
		else
			f = 0.0;
		rsi += arg0Strides.x;

		result[ rdi ] = f;
		rdi += TILE_SIZE;
	}
}