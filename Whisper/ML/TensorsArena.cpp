#include "stdafx.h"
#include "TensorsArena.h"
#include "../D3D/createBuffer.h"
#include "mlUtils.h"

static inline uint32_t roundUpPower2( uint32_t x )
{
	// std::bit_ceil from C++/20 standard library implements runtime dispatch, uses LZCNT when AVX2 is available, otherwise BSR
	// That's not what we want.
	// BSR is only slightly slower than LZCNT: same speed on Intel, on AMD it's 3 versus 1 cycles.
	// defaultNewCapacity function is only called occasionally, that branch is therefore unpredictable.
	assert( x > 1 );
	unsigned long idx;
	_BitScanReverse( &idx, x - 1 );
	return 2u << idx;
}

uint32_t DirectCompute::defaultNewCapacity( uint32_t current, uint32_t requested )
{
	// Implement some reasonable tactics to compute capacity of these buffers

	constexpr uint32_t minAlloc = 1024;	// 1k elements = 4kb of VRAM for FP32 tensors
	constexpr uint32_t allocGranularity = 1u << 14;	// 16k elements = 64kb of VRAM for FP32 tensors

	if( requested > minAlloc )
	{
		const uint32_t roundedUpPowerOf2 = roundUpPower2( requested );

		constexpr uint32_t lowMask = allocGranularity - 1;
		constexpr uint32_t highMask = ~lowMask;
		const uint32_t roundedUpGranularity = ( requested + lowMask ) & highMask;

		const uint32_t res = std::min( roundedUpPowerOf2, roundedUpGranularity );

		assert( res >= requested );
		return res;
	}

	return minAlloc;
}

using namespace DirectCompute;

TensorsArena::ArenaImpl::ArenaImpl( eDataType dataType, const sArenaConfig& config ) :
	type( dataType ),
	pfnNewCap( nullptr != config.pfnCapInner ? config.pfnCapInner : &defaultNewCapacity )
{
	pool.reserve( config.initialCapOuter );
}

Tensor PooledTensor::tensor( eDataType type, const std::array<uint32_t, 4>& ne, pfnNewCapacity pfnNewCap )
{
	const uint32_t p1 = ne[ 0 ] * ne[ 1 ];
	const uint32_t p2 = ne[ 2 ] * ne[ 3 ];
	const uint32_t count = p1 * p2;

	if( count > capacity )
	{
		views.clear();
		const uint32_t newCap = pfnNewCap( capacity, count );
		assert( newCap >= count );

		const size_t cb = elementSize( type ) * newCap;
		CComPtr<ID3D11Buffer> buffer;
		check( createBuffer( eBufferUse::ReadWrite, cb, &buffer, nullptr, nullptr ) );
		check( views.create( buffer, viewFormat( type ), newCap, true ) );
		capacity = newCap;
	}

	TensorShape shape;
	shape.ne = ne;
	shape.setDenseStrides();
	Tensor res{ shape, views };
	res.dbgSetType( type );

#if DBG_TEST_NAN
	fillTensorWithNaN( res );
#endif
	return res;
}

Tensor TensorsArena::ArenaImpl::tensor( const std::array<uint32_t, 4>& ne )
{
	PooledTensor* res;
	if( index >= pool.size() )
	{
		assert( index == pool.size() );
		res = &pool.emplace_back();
	}
	else
		res = &pool[ index ];

	index++;
	return res->tensor( type, ne, pfnNewCap );
}

TensorsArena::TensorsArena( const sArenaConfigs& configs ) :
	arenas{ ArenaImpl{ eDataType::FP16, configs.fp16 }, ArenaImpl{ eDataType::FP32, configs.fp32 } }
{
	static_assert( 0 == (uint8_t)eDataType::FP16 );
	static_assert( 1 == (uint8_t)eDataType::FP32 );
}

Tensor TensorsArena::tensor( eDataType type, const std::array<uint32_t, 4>& ne )
{
	ArenaImpl& arena = arenas[ (uint8_t)type ];
	return arena.tensor( ne );
}

void TensorsArena::reset()
{
	for( ArenaImpl& a : arenas )
		a.reset();
}

void TensorsArena::clear()
{
	for( ArenaImpl& a : arenas )
		a.clear();
}

__m128i TensorsArena::ArenaImpl::getMemoryUse() const
{
	const size_t cbElement = elementSize( type );
	size_t countElts = 0;
	for( const auto& t : pool )
		countElts += t.getCapacity();

	const size_t cbVideo = cbElement * countElts;
	const size_t cbSystem = vectorMemoryUse( pool );
	return setr_size( cbSystem, cbVideo );
}

__m128i TensorsArena::getMemoryUse() const
{
	__m128i res = _mm_setzero_si128();
	for( const auto& a : arenas )
		res = _mm_add_epi64( res, a.getMemoryUse() );
	return res;
}

HRESULT PooledTensor::zeroMemory()
{
	if( 0 == capacity )
		return S_FALSE;
	try
	{
		DirectCompute::zeroMemory( views, capacity );
		return S_OK;
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}

HRESULT TensorsArena::ArenaImpl::zeroMemory()
{
	for( PooledTensor& e : pool )
		CHECK( e.zeroMemory() );
	return S_OK;
}

HRESULT TensorsArena::zeroMemory()
{
	for( ArenaImpl& e : arenas )
		CHECK( e.zeroMemory() );
	return S_OK;
}