// matrix*row vector product, needs first argument reshaped into a sequence of horizontal column major panels
#ifndef TILE_SIZE
static const uint TILE_SIZE = 32;
#endif
#ifndef TILE_HEIGHT
static const uint TILE_HEIGHT = 32;
#endif
#ifndef THREADS_Y
static const uint THREADS_Y = 16;
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
	uint4 arg1Strides: packoffset( c3 );
	uint4 resultSize: packoffset( c4 );
	uint4 resultStrides: packoffset( c5 );
}

groupshared float tileOutput[ THREADS_Y ][ TILE_SIZE ];
groupshared float tile0[ TILE_HEIGHT ][ TILE_SIZE ];
groupshared float tile1[ TILE_HEIGHT ];

void multiplyTiles( const uint3 thread )
{
	float r = 0.0;
	for( uint i = thread.y; i < TILE_HEIGHT; i += THREADS_Y )
	{
		float a = tile0[ i ][ thread.x ];
		float b = tile1[ i ];
		r = mad( a, b, r );
	}
	tileOutput[ thread.y ][ thread.x ] += r;
}

void reduceOutput( const uint3 thread )
{
	float curr = 0.0;
	[branch]
	if( thread.y < THREADS_Y / 2 )
		curr = tileOutput[ thread.y ][ thread.x ];

	for( uint i = THREADS_Y / 2; i > 0; i /= 2 )
	{
		[branch]
		if( thread.y < i )
		{
			curr += tileOutput[ thread.y + i ][ thread.x ];
			tileOutput[ thread.y ][ thread.x ] = curr;
		}
		GroupMemoryBarrierWithGroupSync();
	}
}

void storeTile( const uint threadFlat, const uint4 pos, const uint size )
{
	if( threadFlat >= size )
		return;
	const uint4 prod4 = pos * resultStrides;
	const uint2 prod2 = prod4.xy + prod4.zw;
	uint rdi = prod2.x + prod2.y;
	result[ rdi + threadFlat ] = tileOutput[ 0 ][ threadFlat ];
}

[ numthreads( TILE_SIZE, THREADS_Y, 1 ) ]
void main( const uint3 group: SV_GroupID, const uint3 thread : SV_GroupThreadID, uint threadFlat : SV_GroupIndex )
{
	uint i;
	// Zero all 3 shared buffers
	tileOutput[ thread.y ][ thread.x ] = 0.0;
	for( i = thread.y; i < TILE_HEIGHT; i += THREADS_Y )
		tile0[ i ][ thread.x ] = 0.0;
	if( threadFlat < THREADS_Y )
		tile1[ threadFlat ] = 0.0;

	const uint2 layer = group.yz;
	uint rsi0 = group.x * arg0panel + layer.x * arg0LayerStrides.x + layer.y * arg0LayerStrides.y;
	uint rsi1 = layer.x * arg1Strides.z + layer.y * arg1Strides.w;

	const uint threadOffset = thread.y * TILE_SIZE + thread.x;
	rsi0 += threadOffset;
	rsi1 += threadFlat * arg1Strides.x;

	const uint completeTiles = arg0Size.x / TILE_HEIGHT;
	for( i = 0; i < completeTiles; i++ )
	{
		// Load [ TILE_SIZE, TILE_HEIGHT ] block from the first source tensor into the groupshared buffer
		for( uint j = thread.y; j < TILE_HEIGHT; j += THREADS_Y )
		{
			tile0[ j ][ thread.x ] = arg0[ rsi0 ];
			rsi0 += THREADS_Y * TILE_SIZE;
		}
		// Load [ TILE_HEIGHT ] row from the second source into another groupshared buffer
		[ branch ]
		if( threadFlat < TILE_HEIGHT )
			tile1[ threadFlat ] = arg1[ rsi1 ];
		rsi1 += TILE_HEIGHT * arg1Strides.x;

		GroupMemoryBarrierWithGroupSync();

		multiplyTiles( thread );

		GroupMemoryBarrierWithGroupSync();
	}

	const uint rem = arg0Size.x % TILE_HEIGHT;
	if( rem != 0 )
	{
		for( uint j = thread.y; j < TILE_HEIGHT; j += THREADS_Y )
		{
			float a;
			[branch]
			if( j < rem )
			{
				a = arg0[ rsi0 ];
				rsi0 += THREADS_Y * TILE_SIZE;
			}
			else
				a = 0.0;
			tile0[ j ][ thread.x ] = a;
		}

		if( threadFlat < TILE_HEIGHT )
		{
			float b;
			[branch]
			if( threadFlat < rem )
				b = arg1[ rsi1 ];
			else
				b = 0.0;
			tile1[ threadFlat ] = b;
		}

		GroupMemoryBarrierWithGroupSync();

		multiplyTiles( thread );

		GroupMemoryBarrierWithGroupSync();
	}

	reduceOutput( thread );

	const uint resultPos = group.x * TILE_SIZE;
	const uint outputSize = min( TILE_SIZE, resultSize.x - resultPos );
	storeTile( threadFlat, uint4( resultPos, 0, layer ), outputSize );
}