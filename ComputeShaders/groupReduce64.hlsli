groupshared float sharedAccumulators[ 64 ];

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
void horizontalSum( const uint thread, inout float sum )
{
	sharedAccumulators[ thread ] = sum;
	for( uint i = 32; i > 1; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			sum += sharedAccumulators[ thread + i ];
			sharedAccumulators[ thread ] = sum;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	if( 0 == thread )
		sum += sharedAccumulators[ 1 ];
}

// Compute horisontal sum of the numbers, and broadcast to all threads of the group.
void horizontalSumBroadcast( const uint thread, inout float sum )
{
	horizontalSum( thread, sum );
	if( 0 == thread )
		sharedAccumulators[ 0 ] = sum;
	GroupMemoryBarrierWithGroupSync();
	sum = sharedAccumulators[ 0 ];
}

// Compute horizontal maximum of the numbers, and broadcast to all threads of the group.
void horizontalMaxBroadcast( const uint thread, inout float ax )
{
	sharedAccumulators[ thread ] = ax;
	for( uint i = 32; i > 0; i /= 2 )
	{
		GroupMemoryBarrierWithGroupSync();
		if( thread < i )
		{
			ax = max( ax, sharedAccumulators[ thread + i ] );
			sharedAccumulators[ thread ] = ax;
		}
	}
	GroupMemoryBarrierWithGroupSync();
	ax = sharedAccumulators[ 0 ];
}