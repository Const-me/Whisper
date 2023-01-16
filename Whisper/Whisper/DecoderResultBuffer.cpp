#include "stdafx.h"
#include "DecoderResultBuffer.h"
#include "../D3D/MappedResource.h"
using namespace DirectCompute;

void DecoderResultBuffer::copyFromVram( const Tensor& rsi )
{
	ID3D11ShaderResourceView* srv = rsi;
	if( nullptr == srv )
		throw OLE_E_BLANK;
	if( !rsi.isContinuous() )
		throw E_INVALIDARG;

	const uint32_t len = rsi.countElements();
	if( len > m_capacity )
	{
		buffer = nullptr;
		CD3D11_BUFFER_DESC desc{ len * 4, 0, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ };
		check( device()->CreateBuffer( &desc, nullptr, &buffer ) );
		m_capacity = len;
	}

	CComPtr<ID3D11Resource> source;
	srv->GetResource( &source );
	// Coordinates of a box are in bytes for buffers
	D3D11_BOX box;
	store16( &box, _mm_setr_epi32( 0, 0, 0, (int)( len * 4 ) ) );
	*(uint64_t*)&box.bottom = 0x100000001ull;
	context()->CopySubresourceRegion( buffer, 0, 0, 0, 0, source, 0, &box );
	m_size = len;
}

void DecoderResultBuffer::copyToVector( std::vector<float>& vec ) const
{
	vec.resize( m_size );
	if( vec.empty() )
		throw OLE_E_BLANK;

	MappedResource mapped;
	check( mapped.map( buffer, true ) );
	memcpy( vec.data(), mapped.data(), (size_t)4 * m_size );
}

void DecoderResultBuffer::clear()
{
	buffer = nullptr;
	m_size = m_capacity = 0;
}