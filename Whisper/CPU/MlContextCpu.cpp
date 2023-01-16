#include "stdafx.h"
#include "MlContext.h"
#include "simdUtils.h"
#include "mulMat.h"
using namespace CpuCompute;

MlContext::MlContext( int threads ) : pfor( threads )
{
}

Tensor MlContext::createTensor( eDataType type, const std::array<uint32_t, 4>& size )
{
	Tensor res;
	check( res.create( type, size, allocator ) );
	return res;
}

Tensor MlContext::createTensor( eDataType type, std::initializer_list<uint32_t> size )
{
	Tensor res;
	check( res.create( type, size, allocator ) );
	return res;
}

namespace
{
	inline const uint16_t* getRow16( const Tensor& t, size_t index )
	{
		const uint16_t* rsi = t.fp16();
		rsi += index * t.nb[ 1 ];
		return rsi;
	}
	inline const float* getRow32( const Tensor& t, size_t index )
	{
		const float* rsi = t.fp32();
		rsi += index * t.nb[ 1 ];
		return rsi;
	}
}

Tensor MlContext::addRows( const Tensor& d_te, const Tensor& d_pe, const int* tokens, const int n_tokens, const int n_past )
{
	if( d_te.type() != eDataType::FP16 || d_pe.type() != eDataType::FP32 )
		throw E_INVALIDARG;
	if( d_te.ne[ 0 ] != d_pe.ne[ 0 ] )
		throw E_INVALIDARG;
	if( n_tokens <= 0 )
		throw E_BOUNDS;

	Tensor res = createTensor( eDataType::FP32, { d_te.ne[ 0 ], (uint32_t)n_tokens } );

	const size_t inner = (size_t)d_te.ne[ 0 ];
	const size_t outer = (size_t)n_tokens;
	float* rdi = res.fp32();
	for( size_t i = 0; i < outer; i++, rdi += inner, tokens++ )
	{
		const uint16_t* const source1 = getRow16( d_te, *(const uint32_t*)tokens );
		const float* const source2 = getRow32( d_pe, i + (size_t)n_past );
		addF16to32( rdi, source1, source2, inner );
	}
	return res;
}

namespace
{
	class DispatchHelper3
	{
		std::array<uint32_t, 3> ne;

	public:
		DispatchHelper3() = default;
		DispatchHelper3( uint32_t x, uint32_t y, uint32_t z )
		{
			assert( x > 0 && y > 0 && z > 0 );
			ne[ 0 ] = x;
			ne[ 1 ] = y;
			ne[ 2 ] = z;
		}
		size_t groupsCount() const
		{
			size_t res = ne[ 0 ];
			res *= ne[ 1 ];
			res *= ne[ 2 ];
			return res;
		}
		std::array<uint32_t, 3> unpack( size_t idx ) const
		{
			assert( idx < groupsCount() );
			std::array<uint32_t, 3> res;
			res[ 0 ] = (uint32_t)( idx % ne[ 0 ] );
			idx = idx / ne[ 0 ];
			res[ 1 ] = (uint32_t)( idx % ne[ 1 ] );
			res[ 2 ] = (uint32_t)( idx / ne[ 1 ] );
			return res;
		}
		void next( std::array<uint32_t, 3>& i ) const
		{
			i[ 0 ]++;
			if( i[ 0 ] < ne[ 0 ] )
				return;
			i[ 0 ] = 0;
			i[ 1 ]++;
			if( i[ 1 ] < ne[ 1 ] )
				return;
			i[ 1 ] = 0;
			i[ 2 ]++;
		}
	};

	inline const float* sourceRow( const float* rsi, const std::array<uint32_t, 3>& idx, size_t nb0, size_t nb1, size_t nb2 )
	{
		const size_t r0 = idx[ 0 ] * nb0;
		const size_t r1 = idx[ 1 ] * nb1;
		const size_t r2 = idx[ 2 ] * nb2;
		rsi = rsi + r0 + r1 + r2;
		return rsi;
	}

	struct NormContext : public iComputeRange
	{
		const float* source;
		float* result;
		size_t inner;
		DispatchHelper3 threads;
		std::array<uint32_t, 3> nbInput;

		HRESULT __stdcall compute( size_t i, size_t end ) const override final
		{
			ALIGNED_SPAN( temp, inner );

			std::array<uint32_t, 3> idx = threads.unpack( i );
			float* rdi = result + i * inner;
			for( ; i < end; i++, rdi += inner, threads.next( idx ) )
			{
				const float* rsi = sourceRow( source, idx, nbInput[ 0 ], nbInput[ 1 ], nbInput[ 2 ] );
				norm( rdi, temp, rsi, inner );
			}
			return S_OK;
		}
	};
}

Tensor MlContext::norm( const Tensor& arg )
{
	if( arg.type() != eDataType::FP32 || arg.nb[ 0 ] != 1 )
		throw E_INVALIDARG;
	Tensor res = createTensor( eDataType::FP32, arg.ne );

	NormContext context;
	context.source = arg.fp32();
	context.result = res.fp32();
	context.inner = arg.ne[ 0 ];
	context.threads = DispatchHelper3( arg.ne[ 1 ], arg.ne[ 2 ], arg.ne[ 3 ] );
	context.nbInput = { arg.nb[ 1 ], arg.nb[ 2 ], arg.nb[ 3 ] };

	check( pfor.parallelFor( context, context.threads.groupsCount() ) );
	return res;
}

void MlContext::fmaRepeat( Tensor& cur, const Tensor& w, const Tensor& b )
{
	if( !( cur.isContinuous() && w.isContinuous() && b.isContinuous() ) )
		throw E_INVALIDARG;

	if( !( cur.type() == eDataType::FP32 && w.type() == eDataType::FP32 && b.type() == eDataType::FP32 ) )
		throw E_INVALIDARG;

	if( !isSameShape( w, b ) )
		throw E_INVALIDARG;

	DispatchHelper3 helper{ cur.ne[ 1 ], cur.ne[ 2 ], cur.ne[ 3 ] };
	std::array<uint32_t, 3> idx = { 0, 0, 0 };
	const size_t countRows = helper.groupsCount();

	const size_t innerRes = cur.ne[ 0 ];
	const size_t innerPattern = w.ne[ 0 ];

	float* rdi = cur.fp32();
	for( size_t i = 0; i < countRows; i++, helper.next( idx ), rdi += innerRes )
	{
		std::array<uint32_t, 3> idxPattern;
		idxPattern[ 0 ] = idx[ 0 ] % w.ne[ 1 ];
		idxPattern[ 1 ] = idx[ 1 ] % w.ne[ 2 ];
		idxPattern[ 2 ] = idx[ 2 ] % w.ne[ 3 ];

		const float* s1 = sourceRow( w.fp32(), idxPattern, w.nb[ 1 ], w.nb[ 2 ], w.nb[ 3 ] );
		const float* s2 = sourceRow( b.fp32(), idxPattern, b.nb[ 1 ], b.nb[ 2 ], b.nb[ 3 ] );
		fmaRepeatRow( rdi, innerRes, s1, s2, innerPattern );
	}
}

Tensor MlContext::mulMat( const Tensor& a, const Tensor& b )
{
	if( !DirectCompute::canMulMat( a, b ) )
		throw E_INVALIDARG;

	std::array<uint32_t, 4> ne{ a.ne[ 1 ], b.ne[ 1 ], a.ne[ 2 ], b.ne[ 3 ] };
	Tensor result = createTensor( eDataType::FP32, ne );

	check( CpuCompute::mulMat( result, a, b, pfor ) );
	return result;
}

// cur = add( repeat( b, cur ), cur ); cur = scale(cur, scaling)
void MlContext::addRepeatScale( Tensor& cur, const Tensor& b, float scaling )
{
	if( !( cur.isContinuous() && b.isContinuous() ) )
		throw E_INVALIDARG;
	if( !( cur.type() == eDataType::FP32 && b.type() == eDataType::FP32 ) )
		throw E_INVALIDARG;

	DispatchHelper3 helper{ cur.ne[ 1 ], cur.ne[ 2 ], cur.ne[ 3 ] };
	std::array<uint32_t, 3> idx = { 0, 0, 0 };
	const size_t countRows = helper.groupsCount();

	const size_t innerRes = (uint32_t)cur.ne[ 0 ];
	const size_t innerPattern = (uint32_t)b.ne[ 0 ];

	float* rdi = cur.fp32();
	const __m256 scale = _mm256_set1_ps( scaling );
	for( size_t i = 0; i < countRows; i++, helper.next( idx ), rdi += innerRes )
	{
		std::array<uint32_t, 3> idxPattern;
		idxPattern[ 0 ] = idx[ 0 ] % (uint32_t)b.ne[ 1 ];
		idxPattern[ 1 ] = idx[ 1 ] % (uint32_t)b.ne[ 2 ];
		idxPattern[ 2 ] = idx[ 2 ] % (uint32_t)b.ne[ 3 ];

		const float* source = sourceRow( b.fp32(), idxPattern, b.nb[ 1 ], b.nb[ 2 ], b.nb[ 3 ] );
		addRepeatScaleRow( rdi, innerRes, source, innerPattern, scale );
	}
}

void MlContext::addRepeat( Tensor& cur, const Tensor& b )
{
	if( !( cur.isContinuous() && b.isContinuous() ) )
		throw E_INVALIDARG;
	if( !( cur.type() == eDataType::FP32 && b.type() == eDataType::FP32 ) )
		throw E_INVALIDARG;

	DispatchHelper3 helper{ cur.ne[ 1 ], cur.ne[ 2 ], cur.ne[ 3 ] };
	std::array<uint32_t, 3> idx = { 0, 0, 0 };
	const size_t countRows = helper.groupsCount();

	const size_t innerRes = (uint32_t)cur.ne[ 0 ];
	const size_t innerPattern = (uint32_t)b.ne[ 0 ];

	float* rdi = cur.fp32();
	for( size_t i = 0; i < countRows; i++, helper.next( idx ), rdi += innerRes )
	{
		std::array<uint32_t, 3> idxPattern;
		idxPattern[ 0 ] = idx[ 0 ] % (uint32_t)b.ne[ 1 ];
		idxPattern[ 1 ] = idx[ 1 ] % (uint32_t)b.ne[ 2 ];
		idxPattern[ 2 ] = idx[ 2 ] % (uint32_t)b.ne[ 3 ];

		const float* source = sourceRow( b.fp32(), idxPattern, b.nb[ 1 ], b.nb[ 2 ], b.nb[ 3 ] );
		addRepeatRow( rdi, innerRes, source, innerPattern );
	}
}

// cur = scale(cur, scaling)
void MlContext::scale( Tensor& cur, float scaling )
{
	if( !( cur.isContinuous() && cur.type() == eDataType::FP32 ) )
		throw E_INVALIDARG;

	const size_t len = cur.countElements();
	const __m256 scale = _mm256_set1_ps( scaling );
	scaleRow( cur.fp32(), len, scale );
}

void MlContext::diagMaskInf( Tensor& cur, uint32_t n_past )
{
	if( !( cur.isContinuous() && cur.type() == eDataType::FP32 ) )
		throw E_INVALIDARG;

	const size_t n = cur.countRows();
	const size_t nc = cur.ne[ 0 ];
	const size_t nr = cur.ne[ 1 ];
	const size_t nz = n / nr;

	for( size_t k = 0; k < nz; k++ )
	{
		for( size_t j = 0; j < nr; j++ )
		{
			float* const rdi = cur.fp32() + k * cur.nb[ 2 ] + j * cur.nb[ 1 ];
			// +1 because the original code checked for `if( i > n_past + j )`
			// That's why the first index to write is ( n_past + j + 1 )
			const size_t start = n_past + j + 1;
			const ptrdiff_t len = (ptrdiff_t)nc - (ptrdiff_t)start;
			if( len <= 0 )
				continue;

			// Generates a store string instruction (rep stosd).
			// The magic number is negative infinity in FP32: https://www.h-schmidt.net/FloatConverter/IEEE754.html
			__stosd( (DWORD*)( rdi + start ), 0xff800000u, (size_t)len );
		}
	}
}

void MlContext::softMax( Tensor& cur, float inputScale )
{
	if( !( cur.isContinuous() && cur.type() == eDataType::FP32 ) )
		throw E_INVALIDARG;

	struct SoftMaxContext : public iComputeRange
	{
		float* data;
		float inputScale;
		size_t length, stride;

		HRESULT __stdcall compute( size_t i, size_t end ) const override final
		{
			float* rdi = data + stride * i;
			for( ; i < end; i++, rdi += stride )
				::softMax( rdi, length, inputScale );
			return S_OK;
		}
	};

	SoftMaxContext context;
	context.data = cur.fp32();
	context.inputScale = inputScale;
	context.length = cur.ne[ 0 ];
	context.stride = cur.nb[ 1 ];

	const size_t n = cur.countRows();
	pfor.parallelFor( context, n );
}

namespace
{
	template<class R, class S>
	__forceinline void copyElement( R* rdi, const S* rsi )
	{
		static_assert( std::is_same<R, S>() );
		*rdi = *rsi;
	}
	template<>
	__forceinline void copyElement<float, uint16_t>( float* rdi, const uint16_t* rsi )
	{
		__m128i iv = _mm_cvtsi32_si128( *rsi );
		__m128 fv = _mm_cvtph_ps( iv );
		_mm_store_ss( rdi, fv );
	}
	template<>
	__forceinline void copyElement<uint16_t, float>( uint16_t* rdi, const float* rsi )
	{
		__m128 fv = _mm_load_ss( rsi );
		__m128i iv = _mm_cvtps_ph( fv, 0 );
		*rdi = (uint16_t)(uint32_t)_mm_cvtsi128_si32( iv );
	}

	template<class R, class S>
	__forceinline void copyRow( R* rdi, const S* rsi, size_t length )
	{
		static_assert( std::is_same<R, S>() );
		memcpy( rdi, rsi, length * sizeof( R ) );
	}
	template<>
	__forceinline void copyRow<uint16_t, float>( uint16_t* rdi, const float* rsi, size_t length )
	{
		floatsDowncast( rdi, rsi, length );
	}
	template<>
	__forceinline void copyRow<float, uint16_t>( float* rdi, const uint16_t* rsi, size_t length )
	{
		floatsUpcast( rdi, rsi, length );
	}

	template<class R, class S>
	static void __declspec( noinline ) copyImpl( R* rdi, const S* rsi, const TensorShape& shape )
	{
		const bool continuousRows = shape.nb[ 0 ] == 1;

		for( size_t i03 = 0; i03 < shape.ne[ 3 ]; i03++, rsi += shape.nb[ 3 ] )
		{
			const S* source2 = rsi;
			for( size_t i02 = 0; i02 < shape.ne[ 2 ]; i02++, source2 += shape.nb[ 2 ] )
			{
				const S* source1 = source2;
				for( size_t i01 = 0; i01 < shape.ne[ 1 ]; i01++, source1 += shape.nb[ 1 ] )
				{
					// Performance optimization here: when the rows are dense, we can copy them much faster with memcpy()
					// Or at least with AVX, when we need to convert between numeric types
					if( continuousRows )
					{
						// This branch is very predictable, same outcome for all loop iterations
						copyRow( rdi, source1, shape.ne[ 0 ] );
						rdi += shape.ne[ 0 ];
					}
					else
					{
						const S* source0 = source1;
						for( size_t i00 = 0; i00 < shape.ne[ 0 ]; i00++, source0 += shape.nb[ 0 ] )
						{
							copyElement( rdi, source0 );
							rdi++;
						}
					}
				}
			}
		}
	}
}

HRESULT MlContext::copyImpl( Tensor& result, const Tensor& source )
{
	if( !( result.isContinuous() && ( result.countElements() == source.countElements() ) ) )
		return E_INVALIDARG;

	const eDataType typeResult = result.type();
	const eDataType typeSource = source.type();
	if( source.isContinuous() )
	{
		const size_t elts = result.countElements();
		if( typeResult == typeSource )
		{
			const size_t bytes = elts * elementSize( typeResult );
			memcpy( result.data(), source.data(), bytes );
			return S_OK;
		}
		if( typeSource == eDataType::FP16 && typeResult == eDataType::FP32 )
		{
			floatsUpcast( result.fp32(), source.fp16(), elts );
			return S_OK;
		}
		if( typeSource == eDataType::FP32 && typeResult == eDataType::FP16 )
		{
			floatsDowncast( result.fp16(), source.fp32(), elts );
			return S_OK;
		}
		return E_UNEXPECTED;
	}
	else
	{
		if( typeSource == eDataType::FP16 && typeResult == eDataType::FP16 )
		{
			::copyImpl( result.fp16(), source.fp16(), source );
			return S_OK;
		}
		if( typeSource == eDataType::FP32 && typeResult == eDataType::FP32 )
		{
			::copyImpl( result.fp32(), source.fp32(), source );
			return S_OK;
		}
		if( typeSource == eDataType::FP16 && typeResult == eDataType::FP32 )
		{
			::copyImpl( result.fp32(), source.fp16(), source );
			return S_OK;
		}
		if( typeSource == eDataType::FP32 && typeResult == eDataType::FP16 )
		{
			::copyImpl( result.fp16(), source.fp32(), source );
			return S_OK;
		}
		return E_UNEXPECTED;
	}
}

Tensor MlContext::copy( const Tensor& a, eDataType type, std::initializer_list<uint32_t> size )
{
	const size_t dims = size.size();
	if( 0 == dims || dims > 4 )
		throw E_BOUNDS;

	size_t nRequested = 1;
	for( size_t i = 0; i < dims; i++ )
	{
		uint32_t n = size.begin()[ i ];
		nRequested *= n;
	}
	if( nRequested != a.countElements() )
		throw E_INVALIDARG;

	if( a.type() == type && a.isContinuous() )
	{
		// Same type, and it's dense - no need to move data, equal to reshape
		Tensor res{ a };
		for( size_t i = 0; i < dims; i++ )
			res.ne[ i ] = size.begin()[ i ];;
		for( size_t i = dims; i < 4; i++ )
			res.ne[ i ] = 1;
		res.setDenseStrides();
		return res;
	}
	else
	{
		// Need to convert types, and/or transpose the tensor. Make another tensor for the output
		Tensor res = createTensor( type, size );
		check( copyImpl( res, a ) );
		return res;
	}
}

Tensor MlContext::permute( const Tensor& a, uint8_t axis0, uint8_t axis1, uint8_t axis2, uint8_t axis3 )
{
	assert( axis0 < 4 );
	assert( axis1 < 4 );
	assert( axis2 < 4 );
	assert( axis3 < 4 );

	assert( axis0 != axis1 );
	assert( axis0 != axis2 );
	assert( axis0 != axis3 );
	assert( axis1 != axis2 );
	assert( axis1 != axis3 );
	assert( axis2 != axis3 );

	Tensor res = a;
	res.ne[ axis0 ] = a.ne[ 0 ];
	res.ne[ axis1 ] = a.ne[ 1 ];
	res.ne[ axis2 ] = a.ne[ 2 ];
	res.ne[ axis3 ] = a.ne[ 3 ];

	res.nb[ axis0 ] = a.nb[ 0 ];
	res.nb[ axis1 ] = a.nb[ 1 ];
	res.nb[ axis2 ] = a.nb[ 2 ];
	res.nb[ axis3 ] = a.nb[ 3 ];

	return res;
}

void MlContext::copyInPlace( Tensor& dest, const Tensor& a, eDataType type, std::initializer_list<uint32_t> size )
{
	assert( type == dest.type() );

	const size_t dims = size.size();
	if( 0 == dims || dims > 4 )
		throw E_BOUNDS;

	size_t nRequested = 1;
	for( size_t i = 0; i < dims; i++ )
	{
		uint32_t n = size.begin()[ i ];
		nRequested *= n;
	}
	if( nRequested != a.countElements() || nRequested != dest.countElements() )
		throw E_INVALIDARG;

	// Reshape the destination
	for( size_t i = 0; i < dims; i++ )
		dest.ne[ i ] = size.begin()[ i ];
	for( size_t i = dims; i < 4; i++ )
		dest.ne[ i ] = 1;
	dest.setDenseStrides();

	// Copy the data
	check( copyImpl( dest, a ) );
}

void MlContext::addInPlace( Tensor& a, const Tensor& b )
{
	if( !( a.isContinuous() && b.isContinuous() && a.type() == eDataType::FP32 && b.type() == eDataType::FP32 ) )
		throw E_NOTIMPL;

	const size_t length = a.countElements();
	addRowInPlace( a.fp32(), b.fp32(), length );
}

Tensor MlContext::add( const Tensor& a, const Tensor& b )
{
	if( !( a.isContinuous() && b.isContinuous() && a.type() == eDataType::FP32 && b.type() == eDataType::FP32 ) )
		throw E_NOTIMPL;

	Tensor res = createTensor( eDataType::FP32, a.ne );
	const size_t length = a.countElements();
	addRow( res.fp32(), a.fp32(), b.fp32(), length );
	return res;
}

void MlContext::addRepeatGelu( Tensor& cur, const Tensor& b )
{
	if( !( cur.isContinuous() && b.isContinuous() ) )
		throw E_INVALIDARG;
	if( !( cur.type() == eDataType::FP32 && b.type() == eDataType::FP32 ) )
		throw E_INVALIDARG;

	DispatchHelper3 helper{ cur.ne[ 1 ], cur.ne[ 2 ], cur.ne[ 3 ] };
	std::array<uint32_t, 3> idx = { 0, 0, 0 };
	const size_t countRows = helper.groupsCount();

	const size_t innerRes = (uint32_t)cur.ne[ 0 ];
	const size_t innerPattern = (uint32_t)b.ne[ 0 ];
	float* rdi = cur.fp32();
	auto& lookupTables = getLookupTables();
	for( size_t i = 0; i < countRows; i++, helper.next( idx ), rdi += innerRes )
	{
		std::array<uint32_t, 3> idxPattern;
		idxPattern[ 0 ] = idx[ 0 ] % (uint32_t)b.ne[ 1 ];
		idxPattern[ 1 ] = idx[ 1 ] % (uint32_t)b.ne[ 2 ];
		idxPattern[ 2 ] = idx[ 2 ] % (uint32_t)b.ne[ 3 ];

		const float* source = sourceRow( b.fp32(), idxPattern, b.nb[ 1 ], b.nb[ 2 ], b.nb[ 3 ] );
		addRepeatGeluRow( rdi, innerRes, source, innerPattern, lookupTables );
	}
	return;
}