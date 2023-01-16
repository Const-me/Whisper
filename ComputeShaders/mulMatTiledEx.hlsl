// This compute shader implements yet another version of matrix*matrix product
// For optimal VRAM access pattern, it requires both arguments to be reshaped into a sequence of horizontal column major panels.
// The panel height is TILE_SIZE, and the last panel of the matrix needs to be padded with zeros; see matReshapePanels.hlsl shader for the reshaping.
// So far, it's only used when running on AMD GPUs.
#ifndef TILE_SIZE
static const uint TILE_SIZE = 32;
#endif
#ifndef TILE_HEIGHT
static const uint TILE_HEIGHT = 32;
#endif
#ifndef THREADS_Y
static const uint THREADS_Y = 16;
#endif

#ifndef STREAM_SECOND_MATRIX
#define STREAM_SECOND_MATRIX 1
#endif

// First tensor, reshaped into dense column major horizontal panels of size [ width, TILE_SIZE ]
Buffer<float> arg0: register( t0 );
// Second tensor, reshaped into dense column major horizontal panels of size [ width, TILE_SIZE ]
Buffer<float> arg1: register( t1 );
// FP32 output tensor, row major and continuous
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 arg0Size: packoffset( c0 );
	uint arg0panel: packoffset( c1.y );
	uint2 arg0LayerStrides: packoffset( c1.z );

	// uint4 arg1Size: packoffset( c2 );
	uint arg1panel: packoffset( c3.y );
	uint2 arg1LayerStrides: packoffset( c3.z );

	uint4 resultSize: packoffset( c4 );
	uint4 resultStrides: packoffset( c5 );
}

// Accumulator for the output tile
// That last `+1` helps a bit, I'm not sure why exactly but probebly because memory bank conflicts.
groupshared float tileOutput[ TILE_SIZE ][ TILE_SIZE + 1 ];
// A smaller tile loaded from the first source matrix
groupshared float tile0[ TILE_HEIGHT ][ TILE_SIZE ];
#if !STREAM_SECOND_MATRIX
// A smaller tile loaded from the second source matrix
groupshared float tile1[ TILE_HEIGHT ][ TILE_SIZE ];
#endif

#if STREAM_SECOND_MATRIX
void multiplyTiles( const uint3 thread, uint rsi, const uint h )
{
	uint2 i = uint2( thread.y, rsi );
	const uint2 iInc = uint2( THREADS_Y, THREADS_Y );
	for( ; i.x < TILE_SIZE; i += iInc )
	{
		float r = 0.0;
		uint2 j = uint2( 0, i.y );
		const uint2 jInc = uint2( 1, TILE_SIZE );
		for( ; j.x < h; j += jInc )
		{
			float a = tile0[ j.x ][ thread.x ];
			float b = arg1[ j.y ];
			r = mad( a, b, r );
		}
		tileOutput[ i.x ][ thread.x ] += r;
	}
}
#else
void multiplyTiles( const uint3 thread )
{
	for( uint row = thread.y; row < TILE_SIZE; row += THREADS_Y )
	{
		float r = 0.0;
		for( uint j = 0; j < TILE_HEIGHT; j++ )
		{
			float a = tile0[ j ][ thread.x ];
			float b = tile1[ j ][ row ];
			r = mad( a, b, r );
		}
		tileOutput[ row ][ thread.x ] += r;
	}
}
#endif

void storeTile( const uint3 thread, const uint4 pos, const uint2 size )
{
	if( thread.x >= size.x )
		return;
	const uint4 prod4 = pos * resultStrides;
	const uint2 prod2 = prod4.xy + prod4.zw;
	uint rdi = prod2.x + prod2.y;
	rdi += resultStrides.y * thread.y;
	rdi += thread.x;
	for( uint i = thread.y; i < size.y; i += THREADS_Y, rdi += resultStrides.y * THREADS_Y )
		result[ rdi ] = tileOutput[ i ][ thread.x ];
}

[numthreads( TILE_SIZE, THREADS_Y, 1 )]
void main( const uint3 group: SV_GroupID, const uint3 thread : SV_GroupThreadID )
{
	uint i;
	// Zero all 3 shared buffers
	for( i = thread.y; i < TILE_SIZE; i += THREADS_Y )
		tileOutput[ i ][ thread.x ] = 0.0;
	for( i = thread.y; i < TILE_HEIGHT; i += THREADS_Y )
	{
		tile0[ i ][ thread.x ] = 0.0;
#if !STREAM_SECOND_MATRIX
		tile1[ i ][ thread.x ] = 0.0;
#endif
	}

	const uint2 layer = uint2( group.z % resultSize.z, group.z / resultSize.z );

	uint rsi0 = group.x * arg0panel + layer.x * arg0LayerStrides.x + layer.y * arg0LayerStrides.y;
	uint rsi1 = group.y * arg1panel + layer.x * arg1LayerStrides.x + layer.y * arg1LayerStrides.y;

	const uint threadOffset = thread.y * TILE_SIZE + thread.x;
	rsi0 += threadOffset;
#if STREAM_SECOND_MATRIX
	rsi1 += thread.y;
#else
	rsi1 += threadOffset;
#endif

	const uint completeTiles = arg0Size.x / TILE_HEIGHT;
	for( i = 0; i < completeTiles; i++ )
	{
		// Load [ TILE_SIZE, TILE_HEIGHT ] block from both source tensors into these groupshared buffers
		for( uint j = thread.y; j < TILE_HEIGHT; j += THREADS_Y )
		{
			tile0[ j ][ thread.x ] = arg0[ rsi0 ];
			rsi0 += THREADS_Y * TILE_SIZE;
#if !STREAM_SECOND_MATRIX
			tile1[ j ][ thread.x ] = arg1[ rsi1 ];
			rsi1 += THREADS_Y * TILE_SIZE;
#endif
		}

		// Wait for all threads in the group to complete these loads
		GroupMemoryBarrierWithGroupSync();

#if STREAM_SECOND_MATRIX
		multiplyTiles( thread, rsi1, TILE_HEIGHT );
		rsi1 += TILE_HEIGHT * TILE_SIZE;
#else
		// Multiply + accumulate the elements collected in the groupshared buffers
		multiplyTiles( thread );
#endif
		GroupMemoryBarrierWithGroupSync();
	}

	const uint rem = arg0Size.x % TILE_HEIGHT;
	if( rem != 0 )
	{
		// Load [ TILE_SIZE, rem ] block from both source tensors, and zero out the padding elements
		for( uint j = thread.y; j < TILE_HEIGHT; j += THREADS_Y )
		{
			[branch]
			if( j < rem )
			{
				tile0[ j ][ thread.x ] = arg0[ rsi0 ];
				rsi0 += THREADS_Y * TILE_SIZE;
#if !STREAM_SECOND_MATRIX
				tile1[ j ][ thread.x ] = arg1[ rsi1 ];
				rsi1 += THREADS_Y * TILE_SIZE;
#endif
			}
			else
			{
				tile0[ j ][ thread.x ] = 0.0;
#if !STREAM_SECOND_MATRIX
				tile1[ j ][ thread.x ] = 0.0;
#endif
			}
		}

		// Wait for all threads in the group to complete these loads
		GroupMemoryBarrierWithGroupSync();

		// Multiply + accumulate the elements collected in the groupshared buffers
#if STREAM_SECOND_MATRIX
		multiplyTiles( thread, rsi1, rem );
#else
		multiplyTiles( thread );
#endif
		GroupMemoryBarrierWithGroupSync();
	}

	const uint2 resultPos = group.xy * TILE_SIZE;
	const uint2 outputSize = min( TILE_SIZE, resultSize.xy - resultPos );
	storeTile( thread, uint4( resultPos, layer ), outputSize );
}