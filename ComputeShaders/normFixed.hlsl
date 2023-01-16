// Ported from ggml_compute_forward_norm_f32
// Dispatch [ ne01, ne02, ne03 ] thread groups of this shader
Buffer<float> arg0: register( t0 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	uint4 result_strides: packoffset( c3 );
}

static const float eps = 1e-5f; // TODO: make this a parameter

// #include "groupReduce.hlsli"

#ifndef THREADS
static const uint THREADS = 32;
#endif
static const uint ROW_LENGTH = 1024;
groupshared float rowBuffer[ ROW_LENGTH ];

static const uint REDUCTION_BUFFER = 32;
groupshared float sharedAccumulators[ REDUCTION_BUFFER ];

// Compute horisontal sum of the numbers. The result is only correct on the thread #0 of the group.
void horizontalSum( const uint thread, inout float sum )
{
	if( THREADS > REDUCTION_BUFFER )
	{
		for( uint t = REDUCTION_BUFFER; t < THREADS; t += REDUCTION_BUFFER )
		{
			// Threads [ t .. t + REDUCTION_BUFFER ] store into the buffer
			if( thread >= t && thread < t + REDUCTION_BUFFER )
				sharedAccumulators[ thread - t ] = sum;

			GroupMemoryBarrierWithGroupSync();

			// Threads [ 0 .. REDUCTION_BUFFER ] increment their local sum with the value loaded from the buffer
			if( thread < REDUCTION_BUFFER )
				sum += sharedAccumulators[ thread ];
		}
	}

	if( thread < REDUCTION_BUFFER )
		sharedAccumulators[ thread ] = sum;

	for( uint i = REDUCTION_BUFFER / 2; i > 1; i /= 2 )
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

[ numthreads( THREADS, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint i03 = group.z;
	const uint i02 = group.y;
	const uint i01 = group.x;
	const uint ne00 = ROW_LENGTH;

	// First pass: copy the data to local buffer, and compute sum
	{
		const uint nb01 = src0_strides[ 1 ];
		const uint nb02 = src0_strides[ 2 ];
		const uint nb03 = src0_strides[ 3 ];
		const uint p = i01 * nb01 + i02 * nb02 + i03 * nb03;

		float sum = 0;
		for( uint i = thread; i < ne00; i += THREADS )
		{
			float f = arg0[ p + i ];
			rowBuffer[ i ] = f;
			sum += f;
		}
		horizontalSum( thread, sum );
		if( 0 == thread )
			sharedAccumulators[ 0 ] = sum / (float)(int)ne00;
		GroupMemoryBarrierWithGroupSync();
	}

	// Second pass: offset and compute sum of squares
	{
		const float mean = sharedAccumulators[ 0 ];
		float sum2 = 0;
		for( uint i = thread; i < ne00; i += THREADS )
		{
			float v = rowBuffer[ i ];
			v -= mean;
			rowBuffer[ i ] = v;
			sum2 = mad( v, v, sum2 );
		}
		horizontalSum( thread, sum2 );
		if( 0 == thread )
			sharedAccumulators[ 0 ] = 1.0 / sqrt( sum2 / (float)(int)ne00 + eps );
		GroupMemoryBarrierWithGroupSync();
	}

	// Final pass: apply the scale, and copy from group shared buffer to the destination
	{
		const float scale = sharedAccumulators[ 0 ];

		const uint nb1 = result_strides[ 1 ];
		const uint nb2 = result_strides[ 2 ];
		const uint nb3 = result_strides[ 3 ];
		const uint y = i01 * nb1 + i02 * nb2 + i03 * nb3;

		for( uint i = thread; i < ne00; i += THREADS )
		{
			float v = rowBuffer[ i ];
			v *= scale;
			result[ y + i ] = v;
		}
	}
}