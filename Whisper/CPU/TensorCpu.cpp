#include <stdafx.h>
#include <atomic>
#include "Tensor.h"
using namespace CpuCompute;

#if TENSOR_INTERNAL_ALLOC
namespace
{
	// This structure is immediately before the payload of every tensor which has an internally-allocated memory buffer
	class alignas( 32 ) sTensorMemoryHeader
	{
		std::atomic_ptrdiff_t refCounter;
	public:
		// Reset the counter to the specified value
		void reset( ptrdiff_t rc )
		{
			refCounter = rc;
		}
		// Increment the ref.counter
		void increment()
		{
			refCounter++;
		}
		// Decrement the ref.counter, and return true if it reached zero as the result
		bool decrement()
		{
			ptrdiff_t val = --refCounter;
			assert( val >= 0 );
			return 0 == val;
		}
	};

	inline sTensorMemoryHeader* getMemBlockHeader( void* pv )
	{
		assert( nullptr != pv );
		uint8_t* pb = (uint8_t*)pv;
		static_assert( sizeof( sTensorMemoryHeader ) == 32 );
		return (sTensorMemoryHeader*)( pb - sizeof( sTensorMemoryHeader ) );
	}

	inline void releaseBlock( sTensorMemoryHeader* pointer )
	{
		assert( nullptr != pointer );
		_aligned_free( pointer );
	}

	inline void* allocateBlock( size_t cb, ptrdiff_t initialRefCounter = 1 )
	{
		cb += sizeof( sTensorMemoryHeader );
		void* pv = _aligned_malloc( cb, 32 );
		if( nullptr == pv )
			return nullptr;

		sTensorMemoryHeader* header = (sTensorMemoryHeader*)pv;
		header->reset( initialRefCounter );
		return ( (uint8_t*)pv ) + sizeof( sTensorMemoryHeader );
	}
}

void Tensor::deallocate()
{
	if( ownsMemory && nullptr != m_data )
	{
		sTensorMemoryHeader* const header = getMemBlockHeader( m_data );
		if( header->decrement() )
		{
			// This tensor is the last one which had a reference to that block of memory
			// Release the memory back to the heap
			releaseBlock( header );
		}
	}
	ownsMemory = false;

	TensorShape::setZero();
	m_data = nullptr;
	m_type = (eDataType)0xFF;
}
#endif

Tensor::Tensor( const Tensor& that )
{
	store( ne, that.sizeVec() );
	store( nb, that.stridesVec() );
	m_data = that.m_data;
	m_type = that.m_type;
#if TENSOR_INTERNAL_ALLOC
	if( that.ownsMemory && nullptr != m_data )
	{
		getMemBlockHeader( m_data )->increment();
		ownsMemory = true;
	}
	else
		ownsMemory = false;
#endif
}

Tensor::Tensor( Tensor&& that ) noexcept
{
	store( ne, that.sizeVec() );
	store( nb, that.stridesVec() );
	m_data = that.m_data;
	m_type = that.m_type;
#if TENSOR_INTERNAL_ALLOC
	ownsMemory = that.ownsMemory;
	that.ownsMemory = false;
#endif
	that.m_data = nullptr;
}

void Tensor::operator=( const Tensor& that )
{
	assert( this != &that );
#if TENSOR_INTERNAL_ALLOC
	deallocate();
#endif

	store( ne, that.sizeVec() );
	store( nb, that.stridesVec() );
	m_data = that.m_data;
	m_type = that.m_type;
#if TENSOR_INTERNAL_ALLOC
	if( that.ownsMemory && nullptr != m_data )
	{
		getMemBlockHeader( m_data )->increment();
		ownsMemory = true;
	}
	else
		ownsMemory = false;
#endif
}

void Tensor::operator=( Tensor&& that ) noexcept
{
	assert( this != &that );
#if TENSOR_INTERNAL_ALLOC
	deallocate();
#endif
	store( ne, that.sizeVec() );
	store( nb, that.stridesVec() );
	m_data = that.m_data;
	m_type = that.m_type;
	that.m_data = nullptr;
#if TENSOR_INTERNAL_ALLOC
	ownsMemory = that.ownsMemory;
	that.ownsMemory = false;
#endif
}

HRESULT Tensor::create( eDataType type, const std::array<uint32_t, 4>& sizeElements, iMemoryAllocator* alloc )
{
	const size_t len = (size_t)sizeElements[ 0 ] * sizeElements[ 1 ] * sizeElements[ 2 ] * sizeElements[ 3 ];
	const size_t cbElement = DirectCompute::elementSize( type );
	const size_t cb = len * cbElement;

#if TENSOR_INTERNAL_ALLOC
	deallocate();
#endif

	store( ne, load( sizeElements ) );
	TensorShape::setDenseStrides();
	this->m_type = type;

	if( nullptr != alloc )
	{
#if TENSOR_INTERNAL_ALLOC
		ownsMemory = false;
#endif
		m_data = alloc->allocate( cb, 32 );
		if( nullptr == m_data )
			return E_OUTOFMEMORY;
		return S_OK;
	}
	else
	{
#if TENSOR_INTERNAL_ALLOC
		m_data = allocateBlock( cb, 1 );
		if( nullptr == m_data )
			return E_OUTOFMEMORY;
		ownsMemory = true;
		return S_OK;
#else
		return E_POINTER;
#endif
	}
}

namespace
{
	static HRESULT arrayFromList( std::array<uint32_t, 4>& arr, std::initializer_list<uint32_t> list )
	{
		const size_t dims = list.size();
		if( dims == 0 || dims > 4 )
			return E_INVALIDARG;

		for( size_t i = 0; i < dims; i++ )
		{
			uint32_t u = list.begin()[ i ];
			if( u == 0 )
				return E_INVALIDARG;
			arr[ i ] = u;
		}

		for( size_t i = dims; i < 4; i++ )
			arr[ i ] = 1;

		return S_OK;
	}
}

HRESULT Tensor::create( eDataType type, std::initializer_list<uint32_t> sizeElements, iMemoryAllocator* alloc )
{
	std::array<uint32_t, 4> arr;
	CHECK( arrayFromList( arr, sizeElements ) );

	return create( type, arr, alloc );
}

Tensor::Tensor( void* pointer, eDataType type, std::initializer_list<uint32_t> size )
{
	if( nullptr == pointer )
		throw E_POINTER;
	check( arrayFromList( ne, size ) );
	TensorShape::setDenseStrides();
	m_data = pointer;
	m_type = type;
#if TENSOR_INTERNAL_ALLOC
	ownsMemory = false;
#endif
}

Tensor::Tensor( void* pointer, eDataType type, uint32_t length ) noexcept
{
	// size = [ length, 1, 1, 1 ]
	const __m128i one = _mm_set1_epi32( 1 );
	__m128i v = _mm_insert_epi32( one, (int)length, 0 );
	store( ne, v );
	// stride = [ 1, length, length, length ]
	v = _mm_shuffle_epi32( v, _MM_SHUFFLE( 0, 0, 0, 1 ) );
	store( nb, v );

	m_data = pointer;
	m_type = type;
#if TENSOR_INTERNAL_ALLOC
	ownsMemory = false;
#endif
}

Tensor Tensor::fromData( void* pointer, eDataType type, uint32_t length )
{
	HRESULT hr = E_UNEXPECTED;
	if( nullptr != pointer )
	{
		if( 0 != length )
			return Tensor{ pointer, type, length };
		else
			hr = E_INVALIDARG;
	}
	else
		hr = E_POINTER;
	throw hr;
}

HRESULT Tensor::attach( void* pointer, eDataType type, std::initializer_list<uint32_t> sizeElements )
{
	if( nullptr == pointer )
		return E_POINTER;

	std::array<uint32_t, 4> arr;
	CHECK( arrayFromList( arr, sizeElements ) );

#if TENSOR_INTERNAL_ALLOC
	deallocate();
#endif
	store( ne, load( arr ) );
	TensorShape::setDenseStrides();

	m_data = pointer;
	this->m_type = type;
#if TENSOR_INTERNAL_ALLOC
	ownsMemory = false;
#endif
	return S_OK;
}

Tensor Tensor::reshape3d( uint32_t ne0, uint32_t ne1, uint32_t ne2 ) const
{
	if( !isContinuous() )
		throw E_NOTIMPL;
	if( countElements() != ne0 * ne1 * ne2 )
		throw E_INVALIDARG;

	Tensor res = *this;
	res.ne = { ne0, ne1, ne2, 1 };
	res.setDenseStrides();
	return res;
}

#if TENSOR_GGML_COMPAT
static const __m128i s_maskAlignment16 = _mm_set1_epi64x( 1 );
static const __m128i s_maskAlignment32 = _mm_set1_epi64x( 3 );

bool isAlignedProperly( __m128i r0, __m128i r1, __m128i mask )
{
	__m128i test = _mm_or_si128( r0, r1 );
	return (bool)_mm_testz_si128( test, mask );
}

Tensor::Tensor( const ggml_tensor* ggml )
{
	store( ne, load16( ggml->ne ) );

	__m128i r0 = load16( (const int*)&ggml->nb[ 0 ] );
	__m128i r1 = load16( (const int*)&ggml->nb[ 2 ] );
	// Divide from bytes into elements by right-shifting the 64-bit integers in these vectors
	switch( ggml->type )
	{
	case GGML_TYPE_F16:
		assert( isAlignedProperly( r0, r1, s_maskAlignment16 ) );
		r0 = _mm_srli_epi64( r0, 1 );
		r1 = _mm_srli_epi64( r1, 1 );
		m_type = eDataType::FP16;
		break;

	case GGML_TYPE_F32:
		assert( isAlignedProperly( r0, r1, s_maskAlignment32 ) );
		r0 = _mm_srli_epi64( r0, 2 );
		r1 = _mm_srli_epi64( r1, 2 );
		m_type = eDataType::FP32;
		break;

	case GGML_TYPE_I32:
		assert( isAlignedProperly( r0, r1, s_maskAlignment32 ) );
		r0 = _mm_srli_epi64( r0, 2 );
		r1 = _mm_srli_epi64( r1, 2 );
		m_type = eDataType::U32;
		break;

	default:
		throw E_INVALIDARG;
	}
	// downcast uint64_t into uint32_t in a single vector
	r0 = _mm_shuffle_epi32( r0, _MM_SHUFFLE( 3, 3, 2, 0 ) );
	r1 = _mm_shuffle_epi32( r1, _MM_SHUFFLE( 2, 0, 3, 3 ) );
	store( nb, _mm_blend_epi16( r0, r1, 0b11110000 ) );

	m_data = ggml->data;
}

ggml_tensor Tensor::ggml() const
{
	ggml_tensor res;
	memset( &res, 0, sizeof( ggml_tensor ) );

	const __m128i size = sizeVec();
	store16( res.ne, size );

	const __m128i one = _mm_set1_epi32( 1 );
	const uint32_t maskOnes = (uint32_t)_mm_movemask_ps( _mm_castsi128_ps( _mm_cmpeq_epi32( size, one ) ) );
	const uint32_t maskNotOnes = maskOnes ^ 0b1111;
	unsigned long idx;
	if( _BitScanReverse( &idx, maskNotOnes ) )
		res.n_dims = (int)idx + 1;
	else
		res.n_dims = 0;

	const __m128i strides = stridesVec();
	// Upcast strides from u32 to u64
	const __m128i zero = _mm_setzero_si128();
	__m128i r0 = _mm_unpacklo_epi32( strides, zero );
	__m128i r1 = _mm_unpackhi_epi32( strides, zero );
	// Scale from elements into bytes with left shift vector instructions
	switch( m_type )
	{
	case eDataType::FP16:
		r0 = _mm_slli_epi64( r0, 1 );
		r1 = _mm_slli_epi64( r1, 1 );
		res.type = GGML_TYPE_F16;
		break;
	case eDataType::FP32:
		r0 = _mm_slli_epi64( r0, 2 );
		r1 = _mm_slli_epi64( r1, 2 );
		res.type = GGML_TYPE_F32;
		break;
	case eDataType::U32:
		r0 = _mm_slli_epi64( r0, 2 );
		r1 = _mm_slli_epi64( r1, 2 );
		res.type = GGML_TYPE_I32;
		break;
	default:
		throw OLE_E_BLANK;
	}

	store16( &res.nb[ 0 ], r0 );
	store16( &res.nb[ 2 ], r1 );

	res.data = m_data;
	return res;
}

GgmlTensorView::GgmlTensorView( const Tensor& t ) : tensor( t.ggml() ) {}
#endif