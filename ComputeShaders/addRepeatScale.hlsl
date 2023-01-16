// Compute tensor = ( tensor + repeat( pattern, tensor ) ) * scale in 1 shot, without VRAM allocations
// Dispatch [ nb[ 1 ], nb[ 2 ], nb[ 3 ] ] thread groups of this shader, where nb is size of the destination tensor
RWBuffer<float> tensor: register( u0 );
Buffer<float> pattern: register( t0 );

cbuffer Constants: register( b0 )
{
	uint4 tensorSize: packoffset( c0 );
	uint4 tensorStrides: packoffset( c1 );
	uint4 patternSize: packoffset( c2 );
	uint4 patternStrides: packoffset( c3 );
	float scalingMul : packoffset( c4.x );
}

#ifndef THREADS
#define THREADS 512
#endif

#include "repeatUtils.hlsli"

inline void computeSimple( uint idx, float add )
{
	float f = tensor[ idx ];
	f += add;
	f *= scalingMul;
	tensor[ idx ] = f;
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint3 it = tensorIteratorState( group, thread, tensorSize, tensorStrides );
	uint rsi = rowOffset( group % patternSize.yzw, patternStrides );

	if( patternSize[ 0 ] == 1 )
	{
		// The pattern only has 1 column - broadcasting over the row
		const float p = pattern[ rsi ];
		ROW_LOOP( it )
			computeSimple( it.x, p );
	}
	else if( patternSize[ 0 ] <= THREADS )
	{
		// pattern size doesn't exceed thread group size: load pattern value outside of the loop
		const uint threadsPerGroup = THREADS - ( THREADS % patternSize[ 0 ] );
		if( thread >= threadsPerGroup )
			return;

		const float p = pattern[ rsi + ( thread % patternSize[ 0 ] ) * patternStrides[ 0 ] ];
		ROW_LOOP_EX( it, threadsPerGroup, tensorStrides )
			computeSimple( it.x, p );
	}
	else
	{
		// Pattern rows are larger than the thread group, need to stream from both buffers
		const uint rsiInc = THREADS * patternStrides[ 0 ];
		const uint rsiDec = patternSize[ 0 ] * patternStrides[ 0 ];
		const uint rsiEnd = rsi + rsiDec;
		rsi += thread * patternStrides[ 0 ];

		ROW_LOOP( it )
		{
			float f = tensor[ it.x ];
			float p = pattern[ rsi ];
			rsi += rsiInc;
			if( rsi >= rsiEnd )
				rsi -= rsiDec;
			f += p;
			f *= scalingMul;
			tensor[ it.x ] = f;
		}
	}
}