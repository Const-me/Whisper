#include "stdafx.h"
#include "TempBuffers.h"
#include "../D3D/createBuffer.h"
#include "../D3D/MappedResource.h"
#include "../D3D/shaders.h"
#include "mlUtils.h"
using namespace DirectCompute;

#define CHECK( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) return __hr; }

HRESULT TempBuffers::Buffer::resize( DXGI_FORMAT format, size_t elements, size_t cbElement, bool zeroMemory )
{
	if( elements <= capacity )
	{
		if( zeroMemory )
			TempBuffers::zeroMemory( *this, (uint32_t)elements );
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

void TempBuffers::zeroMemory( ID3D11UnorderedAccessView* uav, uint32_t length )
{
	const __m128i cbData = _mm_cvtsi32_si128( (int)length );
	ID3D11Buffer* cb = updateSmallCb( cbData );

	ID3D11DeviceContext* ctx = context();
	ctx->CSSetUnorderedAccessViews( 0, 1, &uav, nullptr );
	csSetCB( cb );

	constexpr uint32_t THREADS = 512;
	constexpr uint32_t ITERATIONS = 128;
	constexpr uint32_t elementsPerGroup = THREADS * ITERATIONS;
	const uint32_t countGroups = ( length + elementsPerGroup - 1 ) / elementsPerGroup;
	bindShader( eComputeShader::zeroMemory );
	ctx->Dispatch( countGroups, 1, 1 );
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