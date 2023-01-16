inline uint rowOffset( uint3 idx, uint4 strides )
{
	return idx[ 0 ] * strides[ 1 ] + idx[ 1 ] * strides[ 2 ] + idx[ 2 ] * strides[ 3 ];
}

// Initial iterator state for a row of the output tensor
// x = current index, y = index increment, z = end of the index
inline uint3 tensorIteratorState( uint3 group, uint thread, uint4 size, uint4 stride )
{
	uint3 res;
	res.x = rowOffset( group, stride );
	res.y = THREADS * stride[ 0 ];
	res.z = res.x + size[ 0 ] * stride[ 0 ];
	res.x += thread * stride[ 0 ];
	return res;
}

// Handle a complete row of output tensor, using the iterator made by tensorIteratorState() function
#define ROW_LOOP( ts ) for( ; ts.x < ts.z; ts.x += ts.y )
// Same as above, using different row length
#define ROW_LOOP_EX( ts, len, stride ) for( ; ts.x < ts.z; ts.x += len * stride[ 0 ] )