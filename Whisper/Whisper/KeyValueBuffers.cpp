#include "stdafx.h"
#include "KeyValueBuffers.h"
#include "../D3D/createBuffer.h"
#include "../ML/mlUtils.h"
using namespace DirectCompute;

void AttentionBuffer::resize( uint32_t size )
{
	if( size <= m_size )
		return;

	buffer = nullptr;
	check( createBuffer( eBufferUse::ReadWrite, (size_t)2 * size, &buffer, nullptr, nullptr ) );
	m_size = size;
}

Tensor AttentionBuffer::view( uint32_t length, uint32_t offset ) const
{
	if( length + offset > m_size )
		throw E_BOUNDS;
	if( 0 == length )
		throw E_INVALIDARG;

	CComPtr<ID3D11ShaderResourceView> srv;
	CComPtr<ID3D11UnorderedAccessView> uav;

	CD3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{ D3D11_SRV_DIMENSION_BUFFER, DXGI_FORMAT_R16_FLOAT, offset, length };
	check( device()->CreateShaderResourceView( buffer, &srvDesc, &srv ) );

	CD3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{ D3D11_UAV_DIMENSION_BUFFER, DXGI_FORMAT_R16_FLOAT, offset, length };
	check( device()->CreateUnorderedAccessView( buffer, &uavDesc, &uav ) );

	TensorShape shape;
	shape.ne = { length, 1, 1, 1 };
	shape.setDenseStrides();
	return Tensor( shape, srv, uav );
}

void KeyValueBuffers::resize( uint32_t size )
{
	keys.resize( size );
	values.resize( size );
}

HRESULT AttentionBuffer::zeroMemory() const
{
	if( 0 == m_size )
		return S_FALSE;

	CComPtr<ID3D11UnorderedAccessView> uav;
	CD3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{ D3D11_UAV_DIMENSION_BUFFER, DXGI_FORMAT_R16_FLOAT, 0, m_size };
	check( device()->CreateUnorderedAccessView( buffer, &uavDesc, &uav ) );

	try
	{
		DirectCompute::zeroMemory( uav, m_size );
		return S_OK;
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}

HRESULT KeyValueBuffers::zeroMemory() const
{
	CHECK( keys.zeroMemory() );
	CHECK( values.zeroMemory() );
	return S_OK;
}