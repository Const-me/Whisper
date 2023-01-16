// GGML_TASK_COMPUTE step for matrix*matrix product, where nb01 < nb00
Buffer<float> arg0: register( t0 );
Buffer<float> arg1: register( t1 );
RWBuffer<float> resultTensor: register( u0 );
RWBuffer<float> tempBuffer: register( u1 );

cbuffer Constants: register( b0 )
{
	uint4 aSize: packoffset( c0 );
	uint4 aStride: packoffset( c1 );
	uint4 bSize: packoffset( c2 );
	uint4 bStride: packoffset( c3 );
	uint4 resSize: packoffset( c4 );
	bool resultFp16 : packoffset( c5.x );
	uint ne: packoffset( c5.y );
}

#include "miscUtils.hlsli"

// tempBuffer[ rdi .. ] = 0.0
inline void writeTempZeros( uint rdi, const uint len, const uint thread )
{
	const uint rdiEnd = rdi + len;
	for( rdi += thread; rdi < rdiEnd; rdi += 32 )
		tempBuffer[ rdi ] = 0.0;
}

// tempBuffer[ rdi .. ] += mul * arg0[ rsi .. ]
inline void vectorMad( uint rsi, uint rdi, const uint len, const float mul, const uint thread )
{
	const uint rsiEnd = rsi + len;
	rsi += thread;
	rdi += thread;
	for( ; rsi < rsiEnd; rsi += 32, rdi += 32 )
	{
		float f = tempBuffer[ rdi ];
		f = mad( mul, arg0[ rsi ], f );
		[branch]
		if( resultFp16 )
			f = adjustFp16( f );
		tempBuffer[ rdi ] = f;
	}
}

// resultTensor[ rdi .. ] = tempBuffer[ rsi .. ]
inline void copyRow( uint rsi, uint rdi, const uint len, const uint thread )
{
	const uint rsiEnd = rsi + len;
	rsi += thread;
	rdi += thread;
	for( ; rsi < rsiEnd; rsi += 32, rdi += 32 )
	{
		float f = tempBuffer[ rsi ];
		resultTensor[ rdi ] = f;
	}
}

// resultTensor[ rdi .. ] += tempBuffer[ rsi .. ]
inline void addRow( uint rsi, uint rdi, const uint len, const uint thread )
{
	const uint rsiEnd = rsi + len;
	rsi += thread;
	rdi += thread;
	for( ; rsi < rsiEnd; rsi += 32, rdi += 32 )
	{
		float f = resultTensor[ rdi ];
		f += tempBuffer[ rsi ];
		resultTensor[ rdi ] = f;
	}
}

[numthreads( 32, 1, 1 )]
void main( const uint3 group: SV_GroupID, const uint thread : SV_GroupIndex )
{
	const uint i1 = group[ 0 ];
	const uint i2 = group[ 1 ];
	const uint i3 = group[ 2 ];

	const uint ne00 = aSize[ 0 ];
	const uint ne01 = aSize[ 1 ];
	const uint ne02 = aSize[ 2 ];
	const uint ne03 = aSize[ 3 ];

	const uint ne10 = bSize[ 0 ];
	const uint ne11 = bSize[ 1 ];
	const uint ne12 = bSize[ 2 ];
	const uint ne13 = bSize[ 3 ];

	const uint ne0 = resSize[ 0 ];
	const uint ne1 = resSize[ 1 ];
	const uint ne2 = resSize[ 2 ];
	const uint ne3 = resSize[ 3 ];

	const uint nb00 = aStride[ 0 ];
	const uint nb01 = aStride[ 1 ];
	const uint nb02 = aStride[ 2 ];
	const uint nb03 = aStride[ 3 ];

	const uint nb10 = bStride[ 0 ];
	const uint nb11 = bStride[ 1 ];
	const uint nb12 = bStride[ 2 ];
	const uint nb13 = bStride[ 3 ];

	// dst_row = wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0;
	const uint tempRowThread0 = i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0;

	// Faking 4 CPU threads trying to achieve bitwise compatibility with the CPU version
	const uint nth = 4;

	// GGML_TASK_COMPUTE
	{
		// src0_col = src0->data + ( i00 * nb00 + i02 * nb02 + i03 * nb03 );
		const uint aBase = i2 * nb02 + i3 * nb03;
		// src1_val = *      (float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13));
		const uint bBase = i1 * nb11 + i2 * nb12 + i3 * nb13;

		// total columns in src1
		const uint nc = ne10;
		// columns per thread
		const uint dc = ( nc + nth - 1 ) / nth;

		uint tempRow = tempRowThread0;
		for( uint ith = 0; ith < nth; ith++, tempRow += ne )
		{
			writeTempZeros( tempRow, ne01, thread );

			// column range for this thread
			const uint ic0 = dc * ith;
			const uint ic1 = min( ic0 + dc, nc );

			for( uint ic = ic0; ic < ic1; ic++ )
			{
				const uint idxA = aBase + ic * aStride[ 0 ];
				const uint idxB = bBase + ic * bStride[ 0 ];
				const float bValue = arg1[ idxB ];
				vectorMad( idxA, tempRow, ne01, bValue, thread );
			}
		}
	}

	// GGML_TASK_FINALIZE
	{
		const uint rdi = tempRowThread0;
		// const uint rdi = i1 * resSize[ 0 ] + i2 * resSize[ 0 ] * resSize[ 1 ] + i3 * resSize[ 0 ] * resSize[ 1 ] * resSize[ 2 ];
		// const uint rdi = ( ( i3 * resSize[ 2 ] + i2 ) * resSize[ 1 ] + i1 ) * resSize[ 0 ];

		uint tempRow = tempRowThread0;
		copyRow( tempRow, rdi, ne01, thread );

		tempRow += ne;
		for( uint ith = 1; ith < nth; ith++, tempRow += ne )
			addRow( tempRow, rdi, ne01, thread );
	}
}