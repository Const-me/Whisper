// Implementation of fmaRepeat() when both source arguments have same size and strides
// Dispatch [ nb[ 1 ], nb[ 2 ], nb[ 3 ] ] thread groups of this shader, where nb is size of the destination tensor
RWBuffer<float> tensor: register( u0 );
Buffer<float> patternMul: register( t0 );
Buffer<float> patternAdd: register( t1 );

cbuffer Constants: register( b0 )
{
	uint4 tensorSize: packoffset( c0 );
	uint4 tensorStrides: packoffset( c1 );
	uint4 patternSize: packoffset( c2 );
	uint4 patternStrides: packoffset( c3 );
}

#ifndef THREADS
#define THREADS 512
#endif

#include "repeatUtils.hlsli"

inline void computeSimple( uint idx, float mul, float add )
{
	precise float f = tensor[ idx ];
	f *= mul;
	f += add;
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
		const float pMul = patternMul[ rsi ];
		const float pAdd = patternAdd[ rsi ];
		ROW_LOOP( it )
			computeSimple( it.x, pMul, pAdd );
	}
	else if( patternSize[ 0 ] <= THREADS )
	{
		// pattern size doesn't exceed thread group size: load pattern value outside of the loop
		const uint threadsPerGroup = THREADS - ( THREADS % patternSize[ 0 ] );
		if( thread >= threadsPerGroup )
			return;

		rsi += ( thread % patternSize[ 0 ] ) * patternStrides[ 0 ];
		const float pMul = patternMul[ rsi ];
		const float pAdd = patternAdd[ rsi ];
		ROW_LOOP_EX( it, threadsPerGroup, tensorStrides )
			computeSimple( it.x, pMul, pAdd );
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
			precise float f = tensor[ it.x ];
			float mul = patternMul[ rsi ];
			float add = patternAdd[ rsi ];
			rsi += rsiInc;
			if( rsi >= rsiEnd )
				rsi -= rsiDec;
			f *= mul;
			f += add;
			tensor[ it.x ] = f;
		}
	}
}