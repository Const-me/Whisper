#include "stdafx.h"
#include "simdUtils.h"
#include "../ML/LookupTablesData.h"
#include <cmath>
#include <memory>

namespace
{
	constexpr size_t maskAlign8 = ~(size_t)7;

	__forceinline __m256 load8( const uint16_t* rsi )
	{
		__m128i i = _mm_loadu_si128( ( const __m128i* )rsi );
		return _mm256_cvtph_ps( i );
	}

	__forceinline void loadPartial( const uint16_t* x, const uint16_t* y, size_t count, __m256& fx, __m256& fy )
	{
		assert( count < 8 );

		__m128i ix, iy;
		switch( count )
		{
		case 1: // load 2 bytes
			ix = _mm_cvtsi32_si128( *x );
			iy = _mm_cvtsi32_si128( *y );
			break;
		case 2: // load 4 bytes
			ix = _mm_cvtsi32_si128( *(const int*)x );
			iy = _mm_cvtsi32_si128( *(const int*)y );
			break;
		case 3: // load 6 bytes
			ix = _mm_cvtsi32_si128( *(const int*)x );
			iy = _mm_cvtsi32_si128( *(const int*)y );
			ix = _mm_insert_epi16( ix, x[ 2 ], 2 );
			iy = _mm_insert_epi16( iy, y[ 2 ], 2 );
			break;
		case 4: // load 8 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			iy = _mm_cvtsi64_si128( *(const int64_t*)y );
			break;
		case 5: // load 10 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			iy = _mm_cvtsi64_si128( *(const int64_t*)y );
			ix = _mm_insert_epi16( ix, x[ 4 ], 4 );
			iy = _mm_insert_epi16( iy, y[ 4 ], 4 );
			break;
		case 6: // load 12 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			iy = _mm_cvtsi64_si128( *(const int64_t*)y );
			ix = _mm_insert_epi32( ix, *(const int*)( x + 4 ), 2 );
			iy = _mm_insert_epi32( iy, *(const int*)( y + 4 ), 2 );
			break;
		case 7: // load 14 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			iy = _mm_cvtsi64_si128( *(const int64_t*)y );
			ix = _mm_insert_epi32( ix, *(const int*)( x + 4 ), 2 );
			iy = _mm_insert_epi32( iy, *(const int*)( y + 4 ), 2 );
			ix = _mm_insert_epi16( ix, x[ 6 ], 6 );
			iy = _mm_insert_epi16( iy, y[ 6 ], 6 );
			break;
		default:
			fx = fy = _mm256_setzero_ps();
			return;
		}

		fx = _mm256_cvtph_ps( ix );
		fy = _mm256_cvtph_ps( iy );
	}

	__forceinline __m256 loadPartial( const uint16_t* x, size_t count )
	{
		assert( count < 8 );
		__m128i ix;
		switch( count )
		{
		case 1: // load 2 bytes
			ix = _mm_cvtsi32_si128( *x );
			break;
		case 2: // load 4 bytes
			ix = _mm_cvtsi32_si128( *(const int*)x );
			break;
		case 3: // load 6 bytes
			ix = _mm_cvtsi32_si128( *(const int*)x );
			ix = _mm_insert_epi16( ix, x[ 2 ], 2 );
			break;
		case 4: // load 8 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			break;
		case 5: // load 10 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			ix = _mm_insert_epi16( ix, x[ 4 ], 4 );
			break;
		case 6: // load 12 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			ix = _mm_insert_epi32( ix, *(const int*)( x + 4 ), 2 );
			break;
		case 7: // load 14 bytes
			ix = _mm_cvtsi64_si128( *(const int64_t*)x );
			ix = _mm_insert_epi32( ix, *(const int*)( x + 4 ), 2 );
			ix = _mm_insert_epi16( ix, x[ 6 ], 6 );
			break;
		default:
			return _mm256_setzero_ps();
		}
		return  _mm256_cvtph_ps( ix );
	}

	__forceinline __m128 loadFloat2( const float* rsi )
	{
		return _mm_castpd_ps( _mm_load_sd( (const double*)rsi ) );
	}
	__forceinline __m128 loadFloat3( const float* rsi )
	{
		__m128 f = loadFloat2( rsi );
		f = _mm_insert_ps( f, _mm_load_ss( rsi + 2 ), 0x20 );
		return f;
	}

	__forceinline __m256 loadPartial( const float* rsi, size_t count )
	{
		assert( count < 8 );
		__m128 low = _mm_setzero_ps();
		__m128 high = _mm_setzero_ps();
		switch( count )
		{
		case 1:
			low = _mm_load_ss( rsi );
			break;
		case 2:
			low = loadFloat2( rsi );
			break;
		case 3:
			low = loadFloat3( rsi );
			break;
		case 4:
			low = _mm_loadu_ps( rsi );
			break;
		case 5:
			low = _mm_loadu_ps( rsi );
			high = _mm_load_ss( rsi + 4 );
			break;
		case 6:
			low = _mm_loadu_ps( rsi );
			high = loadFloat2( rsi + 4 );
			break;
		case 7:
			low = _mm_loadu_ps( rsi );
			high = loadFloat3( rsi + 4 );
			break;
		}
		return _mm256_setr_m128( low, high );
	}

	__forceinline void storeFloat2( float* rdi, __m128 vec )
	{
		_mm_store_sd( (double*)rdi, _mm_castps_pd( vec ) );
	}

	__forceinline void storePartial( float* rdi, __m256 vec, size_t count )
	{
		assert( count < 8 );

		__m128 tmp = _mm256_castps256_ps128( vec );
		if( count >= 4 )
		{
			_mm_storeu_ps( rdi, tmp );
			if( count == 4 )
				return;
			count -= 4;
			rdi += 4;
			tmp = _mm256_extractf128_ps( vec, 1 );
		}

		switch( count )
		{
		case 1:
			_mm_store_ss( rdi, tmp );
			return;
		case 2:
			storeFloat2( rdi, tmp );
			return;
		case 3:
			storeFloat2( rdi, tmp );
			( (int*)rdi )[ 2 ] = _mm_extract_ps( tmp, 2 );
			return;
		}
	}
}

void addF16to32( float* rdi, const uint16_t* a, const uint16_t* b, size_t length )
{
	const uint16_t* const endAligned = a + ( length & maskAlign8 );
	const size_t rem = length % 8;

	for( ; a < endAligned; a += 8, b += 8, rdi += 8 )
	{
		__m256 f1 = load8( a );
		__m256 f2 = load8( b );
		__m256 res = _mm256_add_ps( f1, f2 );
		_mm256_storeu_ps( rdi, res );
	}

	if( rem != 0 )
	{
		__m256 f1, f2;
		loadPartial( a, b, rem, f1, f2 );
		__m256 res = _mm256_add_ps( f1, f2 );
		storePartial( rdi, res, rem );
	}
}

void addF16to32( float* rdi, const uint16_t* a, const float* b, size_t length )
{
	const uint16_t* const endAligned = a + ( length & maskAlign8 );
	const size_t rem = length % 8;

	for( ; a < endAligned; a += 8, b += 8, rdi += 8 )
	{
		__m256 f1 = load8( a );
		__m256 f2 = _mm256_loadu_ps( b );
		__m256 res = _mm256_add_ps( f1, f2 );
		_mm256_storeu_ps( rdi, res );
	}

	if( rem != 0 )
	{
		__m256 f1 = loadPartial( a, rem );
		__m256 f2 = loadPartial( b, rem );
		__m256 res = _mm256_add_ps( f1, f2 );
		storePartial( rdi, res, rem );
	}
}

alignas( 64 ) const std::array<int, 16> s_zeroTailMask =
{
	-1,-1,-1,-1,-1,-1,-1,-1,
	0, 0, 0, 0, 0, 0, 0, 0,
};

namespace
{
	__forceinline float horizontalSum( __m256 vec )
	{
		__m128 v = _mm256_extractf128_ps( vec, 1 );
		v = _mm_add_ps( v, _mm256_castps256_ps128( vec ) );
		v = _mm_add_ps( v, _mm_movehl_ps( v, v ) );
		v = _mm_add_ss( v, _mm_movehdup_ps( v ) );
		return _mm_cvtss_f32( v );
	}
}

void norm( float* rdi, float* temp, const float* rsi, size_t length )
{
	assert( (size_t)temp % 32 == 0 );
	const float* rsiEndAligned = rsi + ( length & maskAlign8 );
	const size_t rem = length % 8;

	// First pass: copy to temp buffer, and compute the sum; computeVectorSum() in HLSL
	__m256 sum = _mm256_setzero_ps();
	float* t;
	for( t = temp; rsi < rsiEndAligned; rsi += 8, t += 8 )
	{
		__m256 v = _mm256_loadu_ps( rsi );
		sum = _mm256_add_ps( sum, v );
		_mm256_store_ps( t, v );
	}
	float* const tEndAligned = t;
	if( 0 != rem )
	{
		__m256 v = loadPartial( rsi, rem );
		sum = _mm256_add_ps( sum, v );
		_mm256_store_ps( t, v );
		t += 8;
	}

	const float lengthFloat = (float)(int)length;
	const float meanScalar = horizontalSum( sum ) / lengthFloat;
	const __m256 mean = _mm256_set1_ps( meanScalar );

	// Second pass, offsetAndComputeSumSquares() in HLSL
	sum = _mm256_setzero_ps();
	for( t = temp; t < tEndAligned; t += 8 )
	{
		__m256 v = _mm256_load_ps( t );
		v = _mm256_sub_ps( v, mean );
		_mm256_store_ps( t, v );
		sum = _mm256_fmadd_ps( v, v, sum );
	}
	if( 0 != rem )
	{
		__m256 v = _mm256_load_ps( t );
		v = _mm256_sub_ps( v, mean );
		v = _mm256_and_ps( v, loadTailMaskFloats( rem ) );
		_mm256_store_ps( t, v );
		sum = _mm256_fmadd_ps( v, v, sum );
	}

	// Final pass: scale, and copy from temporary buffer into the destination row

	constexpr float eps = 1e-5f; // TODO: make this a parameter
	const float scaleScalar = 1.0f / std::sqrtf( horizontalSum( sum ) / lengthFloat + eps );
	const __m256 scale = _mm256_set1_ps( scaleScalar );

	for( t = temp; t < tEndAligned; t += 8, rdi += 8 )
	{
		__m256 v = _mm256_load_ps( t );
		v = _mm256_mul_ps( v, scale );
		_mm256_storeu_ps( rdi, v );
	}
	if( 0 != rem )
	{
		__m256 v = _mm256_load_ps( t );
		v = _mm256_mul_ps( v, scale );
		storePartial( rdi, v, rem );
	}
}

void fmaRepeatRow( float* rdi, size_t len, const float* w, const float* b, size_t lenPattern )
{
	float* rdiEndAligned = rdi + ( len & maskAlign8 );
	const size_t rem = len % 8;

	if( 1 == lenPattern )
	{
		const __m256 v1 = _mm256_broadcast_ss( w );
		const __m256 v2 = _mm256_broadcast_ss( b );
		for( ; rdi < rdiEndAligned; rdi += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			v = _mm256_fmadd_ps( v, v1, v2 );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			v = _mm256_fmadd_ps( v, v1, v2 );
			_mm256_maskstore_ps( rdi, mask, v );
		}
	}
	else if( len == lenPattern )
	{
		for( ; rdi < rdiEndAligned; rdi += 8, w += 8, b += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			__m256 v1 = _mm256_loadu_ps( w );
			__m256 v2 = _mm256_loadu_ps( b );
			v = _mm256_fmadd_ps( v, v1, v2 );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			__m256 v1 = _mm256_maskload_ps( w, mask );
			__m256 v2 = _mm256_maskload_ps( b, mask );
			v = _mm256_fmadd_ps( v, v1, v2 );
			_mm256_maskstore_ps( rdi, mask, v );
		}
	}
	else
	{
		// TODO: implement if this actually happens
		throw E_NOTIMPL;
	}
}

void __vectorcall addRepeatScaleRow( float* rdi, size_t len, const float* b, size_t lenPattern, const __m256 scale )
{
	float* rdiEndAligned = rdi + ( len & maskAlign8 );
	const size_t rem = len % 8;

	if( 1 == lenPattern )
	{
		const __m256 v2 = _mm256_broadcast_ss( b );
		for( ; rdi < rdiEndAligned; rdi += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			v = _mm256_add_ps( v, v2 );
			v = _mm256_mul_ps( v, scale );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			v = _mm256_add_ps( v, v2 );
			v = _mm256_mul_ps( v, scale );
			_mm256_maskstore_ps( rdi, mask, v );
		}
		return;
	}
	else if( len == lenPattern )
	{
		for( ; rdi < rdiEndAligned; rdi += 8, b += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			__m256 v2 = _mm256_loadu_ps( b );
			v = _mm256_add_ps( v, v2 );
			v = _mm256_mul_ps( v, scale );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			__m256 v2 = _mm256_maskload_ps( b, mask );
			v = _mm256_add_ps( v, v2 );
			v = _mm256_mul_ps( v, scale );
			_mm256_maskstore_ps( rdi, mask, v );
		}
		return;
	}
	else
	{
		// TODO: implement if this actually happens
		throw E_NOTIMPL;
	}
}

void addRepeatRow( float* rdi, size_t len, const float* b, size_t lenPattern )
{
	float* rdiEndAligned = rdi + ( len & maskAlign8 );
	const size_t rem = len % 8;

	if( 1 == lenPattern )
	{
		const __m256 v2 = _mm256_broadcast_ss( b );
		for( ; rdi < rdiEndAligned; rdi += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			v = _mm256_add_ps( v, v2 );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			v = _mm256_add_ps( v, v2 );
			_mm256_maskstore_ps( rdi, mask, v );
		}
		return;
	}
	else if( len == lenPattern )
	{
		for( ; rdi < rdiEndAligned; rdi += 8, b += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			__m256 v2 = _mm256_loadu_ps( b );
			v = _mm256_add_ps( v, v2 );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			__m256 v2 = _mm256_maskload_ps( b, mask );
			v = _mm256_add_ps( v, v2 );
			_mm256_maskstore_ps( rdi, mask, v );
		}
		return;
	}
	else
	{
		// TODO: implement if this actually happens
		throw E_NOTIMPL;
	}
}

namespace
{
	__forceinline __m256 gelu( __m256 x, const DirectCompute::LookupTablesData& lookup )
	{
		__m128i iv = _mm256_cvtps_ph( x, 0 );
		alignas( 16 ) std::array<uint16_t, 8> arr;
		_mm_store_si128( ( __m128i* )arr.data(), iv );
		for( uint16_t& a : arr )
			a = lookup.gelu[ a ];
		iv = _mm_load_si128( ( __m128i* )arr.data() );
		return _mm256_cvtph_ps( iv );
	}
}

void addRepeatGeluRow( float* rdi, size_t len, const float* b, size_t lenPattern, const DirectCompute::LookupTablesData& lookup )
{
	float* rdiEndAligned = rdi + ( len & maskAlign8 );
	const size_t rem = len % 8;

	if( 1 == lenPattern )
	{
		const __m256 v2 = _mm256_broadcast_ss( b );
		for( ; rdi < rdiEndAligned; rdi += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			v = _mm256_add_ps( v, v2 );
			v = gelu( v, lookup );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			v = _mm256_add_ps( v, v2 );
			v = gelu( v, lookup );
			_mm256_maskstore_ps( rdi, mask, v );
		}
		return;
	}
	else if( len == lenPattern )
	{
		for( ; rdi < rdiEndAligned; rdi += 8, b += 8 )
		{
			__m256 v = _mm256_loadu_ps( rdi );
			__m256 v2 = _mm256_loadu_ps( b );
			v = _mm256_add_ps( v, v2 );
			v = gelu( v, lookup );
			_mm256_storeu_ps( rdi, v );
		}
		if( 0 != rem )
		{
			const __m256i mask = loadTailMaskInt( rem );
			__m256 v = _mm256_maskload_ps( rdi, mask );
			__m256 v2 = _mm256_maskload_ps( b, mask );
			v = _mm256_add_ps( v, v2 );
			v = gelu( v, lookup );
			_mm256_maskstore_ps( rdi, mask, v );
		}
		return;
	}
	else
	{
		// TODO: implement if this actually happens
		throw E_NOTIMPL;
	}
}

void __vectorcall scaleRow( float* rdi, size_t len, const __m256 scale )
{
	float* rdiEndAligned = rdi + ( len & maskAlign8 );
	const size_t rem = len % 8;
	for( ; rdi < rdiEndAligned; rdi += 8 )
	{
		__m256 v = _mm256_loadu_ps( rdi );
		v = _mm256_mul_ps( v, scale );
		_mm256_storeu_ps( rdi, v );
	}
	if( 0 != rem )
	{
		const __m256i mask = loadTailMaskInt( rem );
		__m256 v = _mm256_maskload_ps( rdi, mask );
		v = _mm256_mul_ps( v, scale );
		_mm256_maskstore_ps( rdi, mask, v );
	}
}

namespace
{
	using DirectCompute::LookupTablesData;

	__forceinline float horizontalMax( __m256 vec )
	{
		__m128 v = _mm256_extractf128_ps( vec, 1 );
		v = _mm_max_ps( v, _mm256_castps256_ps128( vec ) );
		v = _mm_max_ps( v, _mm_movehl_ps( v, v ) );
		v = _mm_max_ss( v, _mm_movehdup_ps( v ) );
		return _mm_cvtss_f32( v );
	}

	__forceinline float _cvtsh_ss( uint16_t f16 )
	{
		__m128i i = _mm_cvtsi32_si128( f16 );
		__m128 f = _mm_cvtph_ps( i );
		return _mm_cvtss_f32( f );
	}

	__forceinline uint16_t _cvtss_sh( float f, int rounding )
	{
		assert( 0 == rounding );
		__m128 v = _mm_set_ss( f );
		__m128i i = _mm_cvtps_ph( v, 0 );
		return (uint16_t)(uint32_t)_mm_cvtsi128_si32( i );
	}
}

const LookupTablesData& getLookupTables()
{
	static const std::unique_ptr<LookupTablesData> res = std::make_unique<LookupTablesData>();
	return *res;
}

void softMax( float* rdi, size_t length, const float inputScale )
{
	float* const rdiBegin = rdi;
	float* const rdiEndAligned = rdi + ( length & maskAlign8 );
	const size_t remainder = length % 8;
	// First pass, compute maximum
	__m256 max = _mm256_set1_ps( -INFINITY );
	for( rdi = rdiBegin; rdi < rdiEndAligned; rdi += 8 )
	{
		__m256 v = _mm256_loadu_ps( rdi );
		max = _mm256_max_ps( max, v );
	}
	__m256i tailMask;
	if( 0 != remainder )
	{
		tailMask = loadTailMaskInt( remainder );
		__m256 v = _mm256_maskload_ps( rdi, tailMask );
		v = _mm256_max_ps( max, v );
		max = _mm256_blendv_ps( max, v, _mm256_castsi256_ps( tailMask ) );
	}

	// Second pass: apply initial scale, compute the exponent, and compute total sum over the row
	const LookupTablesData& lookup = getLookupTables();
	const float maxScalar = horizontalMax( max );

	float* const rdiEnd = rdiBegin + length;
	double sum = 0;
	for( rdi = rdiBegin; rdi < rdiEnd; rdi++ )
	{
		// Possible to vectorize, but relatively hard
		// An easy way is upcast the complete lookup table to FP32 and then use two _mm256_i32gather_ps instructions per iteration
		// However, that instruction is from AVX2 set. Let's hope this loop won't be a bottleneck.
		float f = *rdi;
		if( f != -INFINITY )
		{
			f = ( f - maxScalar ) * inputScale;
			uint16_t f16 = _cvtss_sh( f, 0 );
			f16 = lookup.exponent[ f16 ];
			f = _cvtsh_ss( f16 );
			sum += f;
		}
		else
			f = 0;

		*rdi = f;
	}

	// Final pass: apply the final scale
	const __m256 finalScale = _mm256_set1_ps( (float)( 1.0 / sum ) );
	for( rdi = rdiBegin; rdi < rdiEndAligned; rdi += 8 )
	{
		__m256 v = _mm256_loadu_ps( rdi );
		v = _mm256_mul_ps( v, finalScale );
		_mm256_storeu_ps( rdi, v );
	}
	if( 0 != remainder )
	{
		__m256 v = _mm256_maskload_ps( rdi, tailMask );
		v = _mm256_mul_ps( v, finalScale );
		_mm256_maskstore_ps( rdi, tailMask, v );
	}
}

void floatsUpcast( float* rdi, const uint16_t* rsi, size_t length )
{
	const uint16_t* rsiEndAligned = rsi + ( length & maskAlign8 );
	const size_t rem = length % 8;

	for( ; rsi < rsiEndAligned; rsi += 8, rdi += 8 )
		_mm256_storeu_ps( rdi, load8( rsi ) );

	if( 0 != rem )
	{
		__m256 v = loadPartial( rsi, rem );
		_mm256_maskstore_ps( rdi, loadTailMaskInt( rem ), v );
	}
}

void floatsDowncast( uint16_t* rdi, const float* rsi, size_t length )
{
	const float* rsiEndAligned = rsi + ( length & maskAlign8 );
	size_t rem = length % 8;

	for( ; rsi < rsiEndAligned; rsi += 8, rdi += 8 )
	{
		__m256 vf = _mm256_loadu_ps( rsi );
		__m128i vi = _mm256_cvtps_ph( vf, 0 );
		store16( rdi, vi );
	}

	if( 0 != rem )
	{
		__m256 vf = _mm256_maskload_ps( rsi, loadTailMaskInt( rem ) );
		__m128i vi = _mm256_cvtps_ph( vf, 0 );
		for( size_t i = 0; i < rem; i++, rdi++ )
		{
			*rdi = (uint16_t)(uint32_t)_mm_cvtsi128_si32( vi );
			vi = _mm_srli_si128( vi, 2 );
		}
	}
}

void addRowInPlace( float* rdi, const float* rsi, size_t length )
{
	const float* rdiEndAligned = rdi + ( length & maskAlign8 );
	size_t rem = length % 8;

	for( ; rdi < rdiEndAligned; rdi += 8, rsi += 8 )
	{
		__m256 a = _mm256_loadu_ps( rdi );
		__m256 b = _mm256_loadu_ps( rsi );
		a = _mm256_add_ps( a, b );
		_mm256_storeu_ps( rdi, a );
	}

	if( 0 != rem )
	{
		const __m256i mask = loadTailMaskInt( rem );
		__m256 a = _mm256_maskload_ps( rdi, mask );
		__m256 b = _mm256_maskload_ps( rsi, mask );
		a = _mm256_add_ps( a, b );
		_mm256_maskstore_ps( rdi, mask, a );
	}
}

void addRow( float* rdi, const float* a, const float* b, size_t length )
{
	const float* aEndAligned = a + ( length & maskAlign8 );
	size_t rem = length % 8;

	for( ; a < aEndAligned; a += 8, b += 8, rdi += 8 )
	{
		__m256 x = _mm256_loadu_ps( a );
		__m256 y = _mm256_loadu_ps( b );
		x = _mm256_add_ps( x, y );
		_mm256_storeu_ps( rdi, x );
	}

	if( 0 != rem )
	{
		const __m256i mask = loadTailMaskInt( rem );
		__m256 x = _mm256_maskload_ps( a, mask );
		__m256 y = _mm256_maskload_ps( b, mask );
		x = _mm256_add_ps( x, y );
		_mm256_maskstore_ps( rdi, mask, x );
	}
}