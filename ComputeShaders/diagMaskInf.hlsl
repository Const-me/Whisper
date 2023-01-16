// ggml_compute_forward_diag_mask_inf_f32
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 elements: packoffset( c0 );
	uint4 strides: packoffset( c1 );
	uint n_past : packoffset( c2.x );
}

static const float negativeInfinity = asfloat( 0xff800000 );

[numthreads( 32, 1, 1 )]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint k = group.y;
	const uint j = group.x;

	// Start of the row
	uint rdi = k * strides[ 2 ] + j * strides[ 1 ];
	// End of the row
	const uint rdiEnd = rdi + elements[ 0 ] * strides[ 0 ];
	// First index to write in this thread
	rdi += ( n_past + j + thread + 1 ) * strides[ 0 ];
	// Index increment
	const uint rdiInc = 32 * strides[ 0 ];

	for( ; rdi < rdiEnd; rdi += rdiInc )
		result[ rdi ] = negativeInfinity;
}