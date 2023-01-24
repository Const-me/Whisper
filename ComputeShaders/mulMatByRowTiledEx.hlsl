// matrix*row vector product, needs first argument reshaped into a sequence of horizontal column major panels
#ifndef TILE_SIZE
static const uint TILE_SIZE = 32;
#endif
#ifndef THREADS_Y
static const uint THREADS_Y = 8;
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

inline uint hadd4( const uint4 v )
{
	const uint2 v2 = v.xy + v.zw;
	return v2.x + v2.y;
}

inline float hadd4( const float4 v )
{
	const float2 v2 = v.xy + v.zw;
	return v2.x + v2.y;
}

groupshared float reductionBuffer[ THREADS_Y ][ TILE_SIZE ];

[numthreads( TILE_SIZE, THREADS_Y, 1 )]
void main( const uint3 group: SV_GroupID, const uint3 thread : SV_GroupThreadID )
{
	const uint2 layer = group.yz;
	// Source offsets for the complete thread group
	uint2 rsi;
	rsi.x = group.x * arg0panel + layer.x * arg0LayerStrides.x + layer.y * arg0LayerStrides.y;
	rsi.y = layer.x * arg1Strides.z + layer.y * arg1Strides.w;
	// Apply source offsets for this particular thread
	rsi.x += thread.y * TILE_SIZE + thread.x;
	rsi.y += thread.y * arg1Strides.x;

	const uint2 rsiInc = uint2( THREADS_Y * TILE_SIZE, THREADS_Y * arg1Strides.x );

	const uint completeTiles = arg0Size.x / ( THREADS_Y * 4 );
	uint i;
	float4 acc = 0.0;
	for( i = 0; i < completeTiles; i++ )
	{
		// Each iteration of this loop consumes THREADS_Y*4 columns from the arg0 panel, and THREADS_Y*4 values from arg1
		float4 v0, v1;
		[unroll]
		for( uint j = 0; j < 4; j++, rsi += rsiInc )
		{
			// Load [ TILE_SIZE, THREADS_Y ] block from the first source tensor
			v0[ j ] = arg0[ rsi.x ];
			// Broadcast [ THREADS_Y ] row from the second source tensor
			v1[ j ] = arg1[ rsi.y ];
		}

		// Now we have [ TILE_SIZE, THREADS_Y * 4 ] block from the first source tensor in the v0 vector,
		// and [ THREADS_Y * 4 ] row from the second one in the v1 vector
		// Multiply and accumulate.
		acc = mad( v0, v1, acc );
	}

	// Handle the remainder columns, if any.
	// When present, their count is in [ 1 .. THREADS_Y * 4 - 1 ] interval
	const uint rem = arg0Size.x % ( THREADS_Y * 4 );
	if( rem != 0 )
	{
		float4 v0 = 0.0, v1 = 0.0;
		[unroll]
		for( uint j = 0; j < 4; j++, rsi += rsiInc )
		{
			const uint x = ( j * THREADS_Y ) + thread.y;
			if( x < rem )
			{
				v0[ j ] = arg0[ rsi.x ];
				v1[ j ] = arg1[ rsi.y ];
			}
		}
		acc = mad( v0, v1, acc );
	}

	// We now have [ TILE_SIZE, THREADS_Y * 4 ] block in the local variables of this thread group
	// The group however only outputs [ TILE_SIZE ] elements max, need a reduction
	float acc1 = hadd4( acc );
	reductionBuffer[ thread.y ][ thread.x ] = acc1;
	GroupMemoryBarrierWithGroupSync();

	for( i = THREADS_Y / 2; i > 1; i /= 2 )
	{
		if( thread.y < i )
		{
			acc1 += reductionBuffer[ thread.y + i ][ thread.x ];
			reductionBuffer[ thread.y ][ thread.x ] = acc1;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if( thread.y != 0 )
		return;

	const uint resultPos = group.x * TILE_SIZE;
	const uint outputSize = min( TILE_SIZE, resultSize.x - resultPos );
	if( thread.x >= outputSize )
		return;

	const uint4 resultPos4 = uint4( resultPos + thread.x, 0, layer );
	const uint rdi = hadd4( resultPos4 * resultStrides );
	result[ rdi ] = acc1 + reductionBuffer[ 1 ][ thread.x ];
}