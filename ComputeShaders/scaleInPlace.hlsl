RWBuffer<float> buffer: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	float multiplier: packoffset( c2.x );
}

[ numthreads( 32, 1, 1 ) ]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint nc0 = src0_elements[ 0 ];
	uint i = group.x * src0_strides[ 1 ];
	const uint iEnd = i + nc0;
	const float mul = multiplier;
	for( i += thread; i < iEnd; i += 32 )
	{
		float f = buffer[ i ];
		f *= mul;
		buffer[ i ] = f;
	}
}