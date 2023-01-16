#include "stdafx.h"
#include "TensorEx.h"
#include "../D3D/createBuffer.h"
#include "../source/ggml.h"
#include "../D3D/MappedResource.h"
using namespace DirectCompute;

HRESULT TensorEx::create( const ggml_tensor& ggml, eBufferUse usage, bool uploadData )
{
	TensorGpuViews::clear();
	buffer = nullptr;
	stagingBuffer = nullptr;

	CHECK( TensorShape::create( ggml ) );
	const ggml_type dataType = ggml.type;
	const uint32_t cbElement = (uint32_t)ggml_type_size( dataType );

	const size_t totalBytes = ggml_nbytes( &ggml );
	if( totalBytes > INT_MAX )
		return DISP_E_OVERFLOW;
	const uint32_t countElements = (uint32_t)( totalBytes / cbElement );

	{
		const void* const rsi = uploadData ? ggml.data : nullptr;
		ID3D11Buffer** ppStagingBuffer = ( usage == eBufferUse::ReadWriteDownload ) ? &stagingBuffer : nullptr;
		CHECK( createBuffer( usage, totalBytes, &buffer, rsi, ppStagingBuffer ) );
	}

	DXGI_FORMAT format;
	switch( dataType )
	{
	case GGML_TYPE_F16:
		format = DXGI_FORMAT_R16_FLOAT;
		break;
	case GGML_TYPE_F32:
		format = DXGI_FORMAT_R32_FLOAT;
		break;
	default:
		return E_NOTIMPL;
	}

	const bool makeUav = usage == eBufferUse::ReadWrite || usage == eBufferUse::ReadWriteDownload;
	return TensorGpuViews::create( buffer, format, totalBytes / cbElement, makeUav );
}

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