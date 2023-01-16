#include "stdafx.h"
#include "TraceStructures.h"
using namespace Tracing;

uint64_t sTraceItem::buffer( uint64_t off, size_t length, eDataType type )
{
	payloadOffset = off;
	payloadSize = length * DirectCompute::elementSize( type );
	*(uint64_t*)( &size[ 0 ] ) = length;
	*(uint64_t*)( &size[ 2 ] ) = 0;
	_mm_storeu_si128( ( __m128i* )stride.data(), _mm_setzero_si128() );
	itemType = eItemType::Buffer;
	dataType = type;
	return payloadSize;
}

uint64_t sTraceItem::tensor( uint64_t off, __m128i ne, __m128i nb, eDataType type )
{
	payloadOffset = off;
	_mm_storeu_si128( ( __m128i* )size.data(), ne );
	_mm_storeu_si128( ( __m128i* )stride.data(), nb );
	uint64_t count = 1;
	for( uint32_t i : size )
		if( i != 0 )
			count *= i;

	payloadSize = count * DirectCompute::elementSize( type );
	itemType = eItemType::Tensor;
	dataType = type;
	return payloadSize;
}