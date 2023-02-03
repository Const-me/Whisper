#include "stdafx.h"
#include "DecoderInputBuffers.h"
#include "../D3D/createBuffer.h"
#include "../D3D/MappedResource.h"
using namespace DirectCompute;

void DecoderInputBuffers::resize( uint32_t size )
{
	if( 0 == size )
		throw E_INVALIDARG;

	if( size <= m_capacity )
	{
		m_size = size;
		return;
	}

	embd = nullptr;

	// Round up by 256, mostly for lulz
	const uint32_t newCapacity = ( size + 0xFFu ) & ( ~( 0xFFu ) );
	const size_t totalBytes = (size_t)4 * newCapacity;

	check( createBuffer( eBufferUse::Dynamic, totalBytes, &embd, nullptr, nullptr ) );

	m_capacity = newCapacity;
	m_size = size;
}

namespace
{
	static Tensor createView( ID3D11Buffer* buffer, uint32_t length )
	{
		Tensor res;

		TensorGpuViews& views = res;
		check( views.create( buffer, DXGI_FORMAT_R32_UINT, length, false ) );

		res.ne = { length, 1, 1, 1 };
		res.setDenseStrides();
		return res;
	}
}

Tensor DecoderInputBuffers::embedding( const int* rsi ) const
{
	if( nullptr == embd || m_size == 0 )
		throw OLE_E_BLANK;

	// Upload the data
	{
		MappedResource mapped;
		check( mapped.map( embd, false ) );
		int* const rdi = (int*)mapped.data();
		memcpy( rdi, rsi, m_size * (size_t)4 );
	}

	return createView( embd, m_size );
}

void DecoderInputBuffers::clear()
{
	embd = nullptr;
	m_size = 0;
	m_capacity = 0;
}

HRESULT DecoderInputBuffers::zeroMemory() const
{
	if( nullptr == embd || m_size == 0 )
		return S_FALSE;

	MappedResource mapped;
	CHECK( mapped.map( embd, false ) );
	__stosd( (DWORD*)mapped.data(), 0, m_capacity );
	return S_OK;
}