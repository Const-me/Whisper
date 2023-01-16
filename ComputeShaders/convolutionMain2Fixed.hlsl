// Optimized version of convolutionMain2.hlsl for kernel size = 3
// Dispatch [ ( ( ne10 / 2 ) + TILE_Y - 1 ) / TILE_Y, ne02, 1 ] thread groups of this shader
#ifndef TILE_Y
static const uint TILE_Y = 8;
#endif
#ifndef THREADS
static const uint THREADS = 64;
#endif

Buffer<float> arg0: register( t0 );
Buffer<float> arg1: register( t1 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	uint4 src1_elements: packoffset( c2 );
	uint4 result_elements: packoffset( c4 );
	uint4 result_strides: packoffset( c5 );
}

// The accumulators we're after
groupshared float resTemp[ TILE_Y ][ THREADS ];

// Multiply + accumulate the specified row
inline void accumulate( float a0, float a1, const uint resultRow, const uint thread )
{
	float acc = resTemp[ resultRow ][ thread ];
	acc = mad( a0, a1, acc );
	resTemp[ resultRow ][ thread ] = acc;
}

inline void convolutionTile( const uint s0, uint s1, const uint thread, const uint stride, const uint height )
{
	// Load 3 rows from arg0
	const float3 a0 = float3( arg0[ s0 ], arg0[ s0 + stride ], arg0[ s0 + stride * 2 ] );

	// Row 0
	float a1 = arg1[ s1 ];
	accumulate( a0[ 0 ], a1, 0, thread );
	s1 += stride;

	for( uint i = 1; i < height; i++ )
	{
		// Row i*2-1
		// Even-indexed rows only contribute to a single output rows, after muiltiplied by kernel row #1
		a1 = arg1[ s1 ];
		accumulate( a0[ 1 ], a1, i - 1, thread );
		s1 += stride;

		// Row i*2, contributes to 2 output rows corresponding to kernel rows #0 and #2
		a1 = arg1[ s1 ];
		accumulate( a0[ 2 ], a1, i - 1, thread );
		accumulate( a0[ 0 ], a1, i, thread );
		s1 += stride;
	}

	// Row height*2 - 1
	a1 = arg1[ s1 ];
	accumulate( a0[ 1 ], a1, height - 1, thread );
	s1 += stride;

	// Row height*2
	a1 = arg1[ s1 ];
	accumulate( a0[ 2 ], a1, height - 1, thread );
}

#include "miscUtils.hlsli"

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint i;
	// Zero out the accumulators
	for( i = 0; i < TILE_Y; i++ )
		resTemp[ i ][ thread ] = 0.0;
	GroupMemoryBarrierWithGroupSync();

	const uint i1 = group.y;
	const uint i0 = group.x * TILE_Y * 2;
	const uint height = min( TILE_Y, ( src1_elements.x / 2 ) - group.x * TILE_Y );

	const uint ne00 = src0_elements[ 0 ];
	const uint ne01 = src0_elements[ 1 ];
	const int ew0 = roundUp32( ne01 );

	uint s0 = i1 * ew0 * ne00;
	const uint s0End = s0 + ew0;
	uint s1 = i0 * ew0;
	s0 += thread;
	s1 += thread;
	for( ; s0 < s0End; s0 += THREADS, s1 += THREADS )
		convolutionTile( s0, s1, thread, ew0, height );

	GroupMemoryBarrierWithGroupSync();

	// Now we need horizontal sums of these shared accumulators, i.e. reduce [height][THREADS] shared array into [height][1] column
	for( i = THREADS / 2; i > 0; i /= 2 )
	{
		if( thread < i )
		{
			for( uint j = 0; j < height; j++ )
			{
				float sum = resTemp[ j ][ thread ];
				sum += resTemp[ j ][ thread + i ];
				resTemp[ j ][ thread ] = sum;
			}
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// And finally, store that column to global memory
	if( thread >= height )
		return;
	const uint nb1 = result_strides[ 1 ];
	const uint rdi = i1 * nb1 + group.x * TILE_Y + thread;
	result[ rdi ] = resTemp[ thread ][ 0 ];
}