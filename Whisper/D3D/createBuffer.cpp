#include "stdafx.h"
#include "createBuffer.h"

#define CHECK( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) return __hr; }

HRESULT DirectCompute::createBuffer( eBufferUse use, size_t totalBytes, ID3D11Buffer** ppGpuBuffer, const void* rsi, ID3D11Buffer** ppStagingBuffer, bool shared )
{
	if( totalBytes > INT_MAX )
		return DISP_E_OVERFLOW;
	if( nullptr == ppGpuBuffer )
		return E_POINTER;

	CD3D11_BUFFER_DESC bufferDesc{ (UINT)totalBytes, D3D11_BIND_SHADER_RESOURCE };
	switch( use )
	{
	case eBufferUse::Immutable:
		if( nullptr == rsi )
			return E_INVALIDARG;
		bufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
		if( gpuInfo().cloneableModel() )
			bufferDesc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
		break;
	case eBufferUse::ReadWrite:
	case eBufferUse::ReadWriteDownload:
		bufferDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
		if( shared && gpuInfo().cloneableModel() )
			bufferDesc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
		break;
	case eBufferUse::Dynamic:
		bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
		bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		break;
	}

	D3D11_SUBRESOURCE_DATA srd;
	D3D11_SUBRESOURCE_DATA* pSrd = nullptr;
	if( nullptr != rsi )
	{
		srd.pSysMem = rsi;
		srd.SysMemPitch = srd.SysMemSlicePitch = 0;
		pSrd = &srd;
	}

	CHECK( device()->CreateBuffer( &bufferDesc, pSrd, ppGpuBuffer ) );

	if( nullptr != ppStagingBuffer && use == eBufferUse::ReadWriteDownload )
	{
		bufferDesc.Usage = D3D11_USAGE_STAGING;
		bufferDesc.BindFlags = 0;
		bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		CHECK( device()->CreateBuffer( &bufferDesc, nullptr, ppStagingBuffer ) );
	}

	return S_OK;
}