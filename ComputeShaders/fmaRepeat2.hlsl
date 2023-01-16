// Implementation of fmaRepeat() when source arguments have different shape or VRAM layout
// Dispatch [ nb[ 1 ], nb[ 2 ], nb[ 3 ] ] thread groups of this shader, where nb is size of the destination tensor
RWBuffer<float> tensor: register( u0 );
Buffer<float> patternMul: register( t0 );
Buffer<float> patternAdd: register( t1 );

cbuffer Constants: register( b0 )
{
	uint4 tensorSize: packoffset( c0 );
	uint4 tensorStrides: packoffset( c1 );
	uint4 patternSizeMul: packoffset( c2 );
	uint4 patternStridesMul: packoffset( c3 );
	uint4 patternSizeAdd: packoffset( c4 );
	uint4 patternStridesAdd: packoffset( c5 );
}

#ifndef THREADS
#define THREADS 32
#endif

#include "repeatUtils.hlsli"

inline float loadPattern( Buffer<float> buffer, uint rowStart, uint i, uint4 size, uint4 stride )
{
	i %= size.x;
	return buffer[ i * stride.x + rowStart ];
}

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	uint3 it = tensorIteratorState( group, thread, tensorSize, tensorStrides );
	const uint rsiMul = rowOffset( group % patternSizeMul.yzw, patternStridesMul );
	const uint rsiAdd = rowOffset( group % patternSizeAdd.yzw, patternStridesAdd );

	for( uint i = thread; it.x < it.z; it.x += it.y, i++ )
	{
		precise float f = tensor[ it.x ];
		float mul = loadPattern( patternMul, rsiMul, i, patternSizeMul, patternStridesMul );
		float add = loadPattern( patternAdd, rsiAdd, i, patternSizeAdd, patternStridesAdd );
		f *= mul;
		f += add;
		tensor[ it.x ] = f;
	}
}