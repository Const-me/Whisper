// An equivalent of "addRepeat.hlsl" followed by "addInPlace.hlsl".
// Merging into a single shader saves some global memory bandwidth and reduces CPU overhead wasted binding resources and dispatching shaders
RWBuffer<float> tensor: register( u0 );
Buffer<float> pattern: register( t0 );
Buffer<float> finalAdd: register( t1 );

cbuffer Constants: register( b0 )
{
	uint4 tensorSize: packoffset( c0 );
	uint4 tensorStrides: packoffset( c1 );
	uint4 patternSize: packoffset( c2 );
	uint4 patternStrides: packoffset( c3 );
	// uint4 finalSize: packoffset( c4 );
	uint4 finalStrides: packoffset( c5 );
}

#ifndef THREADS
#define THREADS 256
#endif

#include "repeatUtils.hlsli"

// The micro-kernel of the shader, computes tensor[ rsi.x ] += pattern + finalAdd[ rsi.y ]
inline void add2( uint2 rsi, float pattern )
{
	float f = tensor[ rsi.x ];
	f += pattern;
	f += finalAdd[ rsi.y ];
	tensor[ rsi.x ] = f;
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint2 stridesX = uint2( tensorStrides.x, finalStrides.x );
	uint2 rsi;
	rsi.x = rowOffset( group, tensorStrides );
	rsi.y = rowOffset( group, finalStrides );
	const uint rsiEnd = rsi.x + tensorSize.x * stridesX.x;
	rsi += stridesX * thread;

	uint pat = rowOffset( group % patternSize.yzw, patternStrides );

	if( patternSize.x == 1 )
	{
		// The pattern only has 1 column, broadcasting over the row
		const uint2 rsiInc = stridesX * THREADS;
		const float p = pattern[ pat ];
		for( ; rsi.x < rsiEnd; rsi += rsiInc )
			add2( rsi, p );
	}
	else if( patternSize.x <= THREADS )
	{
		// pattern size doesn't exceed thread group size, load outside of the loop
		const uint threadsPerGroup = THREADS - ( THREADS % patternSize.x );
		if( thread >= threadsPerGroup )
			return;

		const uint2 rsiInc = stridesX * threadsPerGroup;
		pat += ( thread % patternSize.x ) * patternStrides.x;
		const float p = pattern[ pat ];
		for( ; rsi.x < rsiEnd; rsi += rsiInc )
			add2( rsi, p );
	}
	else
	{
		// Pattern rows are longer than the thread group, need to stream from both buffers
		uint3 rsi3;
		rsi3.xy = rsi;
		rsi3.z = pat + thread * patternStrides.x;

		const uint3 rsiInc = uint3( stridesX, patternStrides.x ) * THREADS;
		while( rsi3.x < rsiEnd )
		{
			add2( rsi3.xy, pattern[ rsi3.z ] );

			rsi3 += rsiInc;
			if( rsi3.z >= patternSize.x )
				rsi3.z -= patternSize.x;
		}
	}
}