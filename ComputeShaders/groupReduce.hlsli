groupshared float sharedAccumulators[ 32 ];

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
void horizontalSum( const uint thread, inout float sum )
{
	sharedAccumulators[ thread ] = sum;
	for( uint i = 16; i > 1; i /= 2 )
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

// Compute horisontal sum of the numbers, in the order equal to the CPU-running dot product implementation.
// The result is only correct on the thread #0 of the group.
void horizontalSumCompat( const uint thread, inout float sum )
{
	sharedAccumulators[ thread ] = sum;
	GroupMemoryBarrierWithGroupSync();

	if( 0 == ( thread & 8 ) )
	{
		// This runs on threads [ 0 .. 7 ] and [ 16 .. 23 ]
		// sum01 = _mm256_add_ps( sum0, sum1 );
		// sum23 = _mm256_add_ps( sum2, sum3 );
		sum += sharedAccumulators[ thread + 8 ];
		sharedAccumulators[ thread ] = sum;
	}

	GroupMemoryBarrierWithGroupSync();
	if( thread < 8 )
	{
		// This runs on threads [ 0 .. 7 ]
		// sum0123 = _mm256_add_ps( sum01, sum23 );
		sum += sharedAccumulators[ thread + 16 ];
		sharedAccumulators[ thread ] = sum;
	}

	GroupMemoryBarrierWithGroupSync();
	if( thread < 4 )
	{
		// const __m128 r4 = _mm_add_ps( _mm256_castps256_ps128( sum0123 ), _mm256_extractf128_ps( sum0123, 1 ) );
		sum += sharedAccumulators[ thread + 4 ];
		sharedAccumulators[ thread ] = sum;
	}

	GroupMemoryBarrierWithGroupSync();
	if( thread < 2 )
	{
		// const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps( r4, r4 ) );
		sum += sharedAccumulators[ thread + 2 ];
		sharedAccumulators[ thread ] = sum;
	}

	GroupMemoryBarrierWithGroupSync();
	if( 0 == thread )
	{
		// const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );
		sum += sharedAccumulators[ 1 ];
	}
}

// Compute horisontal sum of the numbers, in yet another creative summation order recently implemented in the upstream
void horizontalSumCompatNew( const uint thread, inout float sum )
{
	// GGML_F32x8_REDUCE
	sharedAccumulators[ thread ] = sum;
	GroupMemoryBarrierWithGroupSync();

	if( 0 == ( thread & 8 ) )
	{
		// Runs on threads [ 0 .. 7 ] and [ 16 .. 23 ]
		sum += sharedAccumulators[ thread | 8 ];
		sharedAccumulators[ thread ] = sum;
	}
	GroupMemoryBarrierWithGroupSync();

	if( thread < 8 )
	{
		// Runs on threads [ 0 .. 7 ]
		sum += sharedAccumulators[ thread | 0x10 ];
		sharedAccumulators[ thread ] = sum;
	}
	GroupMemoryBarrierWithGroupSync();

	if( thread < 4 )
	{
		// Runs on threads [ 0 .. 3 ]
		sum += sharedAccumulators[ thread | 4 ];
		sharedAccumulators[ thread ] = sum;
	}
	GroupMemoryBarrierWithGroupSync();

	if( thread < 4 && 0 == ( thread & 1 ) )
	{
		// Runs on threads [ 0, 2 ]
		sum += sharedAccumulators[ thread | 1 ];
		sharedAccumulators[ thread ] = sum;
	}
	GroupMemoryBarrierWithGroupSync();

	if( 0 == thread )
		sum += sharedAccumulators[ 2 ];
}


// Compute horizontal maximum of the numbers, and broadcast to all threads of the group.
void horizontalMaxBroadcast( const uint thread, inout float ax )
{
	sharedAccumulators[ thread ] = ax;
	for( uint i = 16; i > 0; i /= 2 )
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