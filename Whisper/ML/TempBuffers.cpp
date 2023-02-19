#include "stdafx.h"
#include "TempBuffers.h"
#include "../D3D/createBuffer.h"
#include "../D3D/MappedResource.h"
#include "mlUtils.h"
using namespace DirectCompute;

#define CHECK( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) return __hr; }

HRESULT TempBuffers::Buffer::resize( DXGI_FORMAT format, size_t elements, size_t cbElement, bool zeroMemory )
{
	if( elements <= capacity )
	{
		if( zeroMemory )
			DirectCompute::zeroMemory( *this, (uint32_t)elements );
		return S_OK;
	}
	clear();

	CComPtr<ID3D11Buffer> buffer;
	const size_t totalBytes = elements * cbElement;
	CHECK( createBuffer( eBufferUse::ReadWrite, totalBytes, &buffer, nullptr, nullptr ) );
	CHECK( TensorGpuViews::create( buffer, format, elements, true ) );
	capacity = elements;
	return S_OK;
}

const TensorGpuViews& TempBuffers::fp16( size_t countElements, bool zeroMemory )
{
	HRESULT hr = m_fp16.resize( DXGI_FORMAT_R16_FLOAT, countElements, 2, zeroMemory );
	if( FAILED( hr ) )
		throw hr;
	return m_fp16;
}

const TensorGpuViews& TempBuffers::fp16_2( size_t countElements, bool zeroMemory )
{
	HRESULT hr = m_fp16_2.resize( DXGI_FORMAT_R16_FLOAT, countElements, 2, zeroMemory );
	if( FAILED( hr ) )
		throw hr;
	return m_fp16_2;
}

const TensorGpuViews& TempBuffers::fp32( size_t countElements, bool zeroMemory )
{
	HRESULT hr = m_fp32.resize( DXGI_FORMAT_R32_FLOAT, countElements, 4, zeroMemory );
	if( FAILED( hr ) )
		throw hr;
	return m_fp32;
}

__m128i TempBuffers::getMemoryUse() const
{
	size_t cb = m_fp16.getCapacity() * 2;
	cb += m_fp16_2.getCapacity() * 2;
	cb += m_fp32.getCapacity() * 4;
	return setHigh_size( cb );
}