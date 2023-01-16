// GGML_TASK_COMPUTE step for matrix*matrix product, where nb01 >= nb00;
// Dispatch with [ ne11, ne01*ne02*ne03 ] thread groups
// Each thread group computes a single dot product
Buffer<float> arg0: register( t0 );
Buffer<float> arg1: register( t1 );
RWBuffer<float> result: register( u0 );

cbuffer Constants: register( b0 )
{
	uint4 src0_elements: packoffset( c0 );
	uint4 src0_strides: packoffset( c1 );
	uint4 src1_elements: packoffset( c2 );
	uint4 result_elements: packoffset( c4 );
	uint4 result_strides: packoffset( c5 );
}

inline uint product( uint3 vec )
{
	return vec.x * vec.y * vec.z;
}

inline uint product( uint4 vec )
{
	uint2 tmp = vec.xy * vec.zw;
	return tmp.x * tmp.y;
}

inline float dotProductInner( uint i0, uint i1, uint length, uint thread )
{
	float res = 0;
	for( uint i = thread; i < length; i += 32 )
		res = mad( arg0[ i0 + i ], arg1[ i1 + i ], res );
	return res;
}

#include "groupReduce.hlsli"

[numthreads( 32, 1, 1 )]
void main( uint3 group: SV_GroupID, uint thread : SV_GroupIndex )
{
	const uint ne00 = src0_elements.x;
	const uint ne01 = src0_elements.y;
	const uint ne02 = src0_elements.z;
	const uint ne03 = src0_elements.w;

	const uint ne10 = src1_elements.x;
	const uint ne11 = src1_elements.y;
	const uint ne12 = src1_elements.z;
	const uint ne13 = src1_elements.w;

	const int nb00 = src0_strides.x;
	const int nb01 = src0_strides.y;
	const int nb02 = src0_strides.z;
	const int nb03 = src0_strides.w;

	// total rows in src0
	// const int nr = ne01*ne02*ne03;
	const uint nr = product( src0_elements.yzw );

	const uint ir = group.y;

	// src0 indices
	const uint i03 = ir / ( ne02 * ne01 );
	const uint i02 = ( ir - i03 * ne02 * ne01 ) / ne01;
	const uint i01 = ( ir - i03 * ne02 * ne01 - i02 * ne01 );

	const uint i13 = i03;
	const uint i12 = i02;

	const uint i0 = i01;
	const uint i2 = i02;
	const uint i3 = i03;

	// src0_row = (ggml_fp16_t *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
	// src1_col = wdata + ( i13 * ne12 * ne11 + i12 * ne11 + 0 ) * ne00;
	const uint src0_row = i01 * nb01 + i02 * nb02 + i03 * nb03;
	const uint src1_col = ( i13 * ne12 * ne11 + i12 * ne11 ) * ne00;

	const uint ic = group.x;
	float curr = dotProductInner( src0_row, src1_col + ic * ne00, ne00, thread );
	horizontalSumCompatNew( thread, curr );

	if( 0 != thread )
		return;

	const uint nb0 = result_strides.x;
	const uint nb1 = result_strides.y;
	const uint nb2 = result_strides.z;
	const uint nb3 = result_strides.w;

	const uint ne0 = result_elements.x;
	// float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));
	const uint dst_col = i0 * nb0 + i2 * nb2 + i3 * nb3;
	result[ dst_col + ic * ne0 ] = curr;
}