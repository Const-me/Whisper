#pragma once
#include <immintrin.h>

void addF16to32( float* rdi, const uint16_t* a, const uint16_t* b, size_t length );
void addF16to32( float* rdi, const uint16_t* a, const float* b, size_t length );

class AlignedSpan
{
	float* pointer;

public:
	AlignedSpan( void* data )
	{
		size_t i = (size_t)data;
		constexpr size_t mask32 = ~(size_t)31;
		i = ( i + 31 ) & mask32;
		pointer = (float*)i;
	}

	operator float* ( ) { return pointer; }
};

inline size_t tempBufferForFloats( size_t count )
{
	// Round up by 8 to be able to use full-vector loads and stores
	constexpr size_t mask8 = ~(size_t)7;
	count = ( count + 7 ) & mask8;

	// Add 32 more bytes to align the temporary buffer
	return ( count * 4 ) + 32;
}

#define ALIGNED_SPAN( name, countFloats ) AlignedSpan name{ _alloca( tempBufferForFloats( countFloats ) ) }

void norm( float* rdi, float* temp, const float* rsi, size_t length );

void fmaRepeatRow( float* rdi, size_t len, const float* w, const float* b, size_t lenPattern );
void __vectorcall addRepeatScaleRow( float* rdi, size_t len, const float* b, size_t lenPattern, const __m256 scale );
void addRepeatRow( float* rdi, size_t len, const float* b, size_t lenPattern );
void __vectorcall scaleRow( float* rdi, size_t len, const __m256 scale );

namespace DirectCompute
{
	struct LookupTablesData;
}
const DirectCompute::LookupTablesData& getLookupTables();
void addRepeatGeluRow( float* rdi, size_t len, const float* b, size_t lenPattern, const DirectCompute::LookupTablesData& lookup );

void softMax( float* rdi, size_t length, const float inputScale );

// A cache line-aligned array where first 8 elements have all bits set, last 8 elements are zeros
extern const std::array<int, 16> s_zeroTailMask;

// Load a tail mask as FP32 vector, for use with _mm256_and_ps or _mm256_blendv_ps instructions
__forceinline __m256 loadTailMaskFloats( size_t remainder )
{
	assert( remainder > 0 && remainder < 8 );
	const float* rsi = (const float*)&s_zeroTailMask;
	rsi += 8;
	return _mm256_loadu_ps( rsi - remainder );
}

// Load a tail mask as int32 vector, for use with _mm256_maskstore_ps instruction
template<bool assertIncomplete = true>
__forceinline __m256i loadTailMaskInt( size_t remainder )
{
	if constexpr( assertIncomplete )
		assert( remainder > 0 && remainder < 8 );
	else
		assert( remainder >= 0 && remainder <= 8 );

	const int* rsi = (const int*)&s_zeroTailMask;
	rsi += 8;
	return _mm256_loadu_si256( ( const __m256i* )( rsi - remainder ) );
}

void floatsUpcast( float* rdi, const uint16_t* rsi, size_t length );

void floatsDowncast( uint16_t* rdi, const float* rsi, size_t length );

void addRowInPlace( float* rdi, const float* rsi, size_t length );
void addRow( float* rdi, const float* a, const float* b, size_t length );