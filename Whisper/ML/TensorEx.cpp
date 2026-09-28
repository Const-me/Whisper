#include "stdafx.h"
#include "TensorEx.h"
#include "../D3D/createBuffer.h"
#include "../source/ggml.h"
#include "../D3D/MappedResource.h"
using namespace DirectCompute;

HRESULT TensorEx::create( eDataType type, eBufferUse usage, const std::array<uint32_t, 4>& sizeElements )
{
	TensorGpuViews::clear();
	buffer = nullptr;
	stagingBuffer = nullptr;
	std::initializer_list<uint32_t> il( sizeElements.data(), sizeElements.data() + 4 );

	ID3D11Buffer** ppStaging = ( usage == eBufferUse::ReadWriteDownload ) ? &stagingBuffer : nullptr;
	return Tensor::create( type, il, usage, buffer, nullptr, ppStaging );
}

HRESULT TensorEx::getViewSize( uint32_t& cbElement, uint32_t& countElements ) const
{
	ID3D11ShaderResourceView* const srv = *this;
	if( nullptr == srv )
		return OLE_E_BLANK;

	D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
	srv->GetDesc( &viewDesc );

	cbElement = dxgiSizeof( viewDesc.Format );

	assert( viewDesc.ViewDimension == D3D_SRV_DIMENSION_BUFFER );
	assert( viewDesc.Buffer.FirstElement == 0 );
	countElements = viewDesc.Buffer.NumElements;

	return S_OK;
}

HRESULT TensorEx::download( void* rdi, size_t cb ) const
{
	if( nullptr == stagingBuffer )
		return HRESULT_FROM_WIN32( ERROR_GPIO_OPERATION_DENIED );	// The requested operation is not supported for the specified handle.

	ID3D11DeviceContext* const ctx = context();
	ctx->CopyResource( stagingBuffer, buffer );

	MappedResource mapped;
	CHECK( mapped.map( stagingBuffer, true ) );
	memcpy( rdi, mapped.data(), cb );

	return S_OK;
}

HRESULT TensorEx::download( void* rdi ) const
{
	uint32_t cbElement, numElements;
	CHECK( getViewSize( cbElement, numElements ) );

	size_t cb = (size_t)cbElement * numElements;
	return download( rdi, cb );
}