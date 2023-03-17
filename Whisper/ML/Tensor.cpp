#include "stdafx.h"
#include "Tensor.h"
#include "../D3D/MappedResource.h"
#include "../D3D/createBuffer.h"
#include "../source/ggml.h"
using namespace DirectCompute;

Tensor::Tensor( const Tensor& that )
{
	ne = that.ne;
	nb = that.nb;
	srv = that.srv;
	uav = that.uav;
#ifdef _DEBUG
	dbgType = that.dbgType;
#endif
}

Tensor::Tensor( Tensor&& that ) noexcept
{
	ne = that.ne;
	nb = that.nb;
	srv.Attach( that.srv.Detach() );
	uav.Attach( that.uav.Detach() );
#ifdef _DEBUG
	dbgType = that.dbgType;
#endif
}

Tensor& Tensor::operator=( const Tensor& that )
{
	ne = that.ne;
	nb = that.nb;
	srv = that.srv;
	uav = that.uav;
#ifdef _DEBUG
	dbgType = that.dbgType;
#endif
	return *this;
}

Tensor& Tensor::operator=( Tensor&& that ) noexcept
{
	ne = that.ne;
	nb = that.nb;
	srv.Attach( that.srv.Detach() );
	uav.Attach( that.uav.Detach() );
#ifdef _DEBUG
	dbgType = that.dbgType;
#endif
	return *this;
}

Tensor::Tensor( const TensorShape& shape, CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11UnorderedAccessView>& uav ) noexcept :
	TensorShape( shape )
{
	TensorGpuViews::srv.Attach( srv.Detach() );
	TensorGpuViews::uav.Attach( uav.Detach() );
}

Tensor::Tensor( const TensorShape& shape, const TensorGpuViews& views ) :
	TensorShape( shape )
{
	srv = views;
	uav = views;
}

HRESULT Tensor::create( const ggml_tensor& ggml, eBufferUse usage, bool uploadData )
{
	TensorGpuViews::clear();

	switch( usage )
	{
	case eBufferUse::Immutable:
	case eBufferUse::ReadWriteDownload:
		break;
	default:
		return E_INVALIDARG;
	}

	CComPtr<ID3D11Buffer> buffer;

	CHECK( TensorShape::create( ggml ) );
	const ggml_type dataType = ggml.type;
	const uint32_t cbElement = (uint32_t)ggml_type_size( dataType );

	const size_t totalBytes = ggml_nbytes( &ggml );
	if( totalBytes > INT_MAX )
		return DISP_E_OVERFLOW;
	const uint32_t countElements = (uint32_t)( totalBytes / cbElement );

	{
		const void* const rsi = uploadData ? ggml.data : nullptr;
		CHECK( createBuffer( usage, totalBytes, &buffer, rsi, nullptr ) );
	}

	DXGI_FORMAT format;
	eDataType type;
	switch( dataType )
	{
	case GGML_TYPE_F16:
		format = DXGI_FORMAT_R16_FLOAT;
		type = eDataType::FP16;
		break;
	case GGML_TYPE_F32:
		format = DXGI_FORMAT_R32_FLOAT;
		type = eDataType::FP32;
		break;
	default:
		return E_NOTIMPL;
	}

	const bool makeUav = ( usage == eBufferUse::ReadWrite );

	CHECK( TensorGpuViews::create( buffer, format, totalBytes / cbElement, makeUav ) );
#ifdef _DEBUG
	dbgType.type = type;
	dbgType.usage = usage;
	dbgType.hasInitialData = uploadData;
#endif
	return S_OK;
}

HRESULT Tensor::createImmutable( eDataType type, const std::array<int, 4>& size, const void* rsi )
{
	size_t elts = (uint32_t)size[ 0 ];
	elts *= (uint32_t)size[ 1 ];
	elts *= (uint32_t)size[ 2 ];
	elts *= (uint32_t)size[ 3 ];

	DXGI_FORMAT format;
	size_t cbElement;
	switch( type )
	{
	case eDataType::FP16:
		format = DXGI_FORMAT_R16_FLOAT;
		cbElement = 2;
		break;
	case eDataType::FP32:
		format = DXGI_FORMAT_R32_FLOAT;
		cbElement = 4;
		break;
	default:
		return E_NOTIMPL;
	}

	CComPtr<ID3D11Buffer> buffer;
	CHECK( createBuffer( eBufferUse::Immutable, cbElement * elts, &buffer, rsi, nullptr ) );
	CHECK( TensorGpuViews::create( buffer, format, elts, false ) );

	__m128i v = _mm_loadu_si128( ( const __m128i* )size.data() );
	_mm_storeu_si128( ( __m128i* )ne.data(), v );
	setDenseStrides();
	return S_OK;
}

HRESULT Tensor::create( eDataType type, std::initializer_list<uint32_t> sizeElements, eBufferUse usage, CComPtr<ID3D11Buffer>& buffer, const void* rsi, ID3D11Buffer** ppStagingBuffer, bool shared )
{
	TensorGpuViews::clear();

	size_t nDims = sizeElements.size();
	if( 0 == nDims || nDims > 4 )
		return E_INVALIDARG;
	nDims = std::min( nDims, (size_t)4 );
	size_t totalElements = 1;
	for( size_t i = 0; i < nDims; i++ )
	{
		uint32_t n = sizeElements.begin()[ i ];
		if( n == 0 )
			return E_INVALIDARG;
		ne[ i ] = n;
		totalElements *= n;
	}

	DXGI_FORMAT format;
	size_t cbElement;
	switch( type )
	{
	case eDataType::FP32:
		format = DXGI_FORMAT_R32_FLOAT;
		cbElement = 4;
		break;
	case eDataType::FP16:
		format = DXGI_FORMAT_R16_FLOAT;
		cbElement = 2;
		break;
	case eDataType::U32:
		format = DXGI_FORMAT_R32_UINT;
		cbElement = 4;
		break;
	default:
		return E_NOTIMPL;
	}

	const size_t totalBytes = cbElement * totalElements;
	if( totalBytes > INT_MAX )
		return DISP_E_OVERFLOW;

	for( size_t i = nDims; i < 4; i++ )
		ne[ i ] = 1;
	TensorShape::setDenseStrides();

	CHECK( createBuffer( usage, totalBytes, &buffer, rsi, ppStagingBuffer, shared ) );

	CHECK( TensorGpuViews::create( buffer, format, totalBytes / cbElement, true ) );
#ifdef _DEBUG
	dbgType.type = type;
	dbgType.usage = usage;
	dbgType.hasInitialData = ( nullptr != rsi );
#endif
	return S_OK;
}

HRESULT Tensor::create( eDataType type, std::initializer_list<uint32_t> sizeElements, bool shared )
{
	CComPtr<ID3D11Buffer> buffer;
	return create( type, sizeElements, eBufferUse::ReadWrite, buffer, nullptr, nullptr, shared );
}

HRESULT Tensor::create( eDataType type, const std::array<uint32_t, 4>& sizeElements, bool shared )
{
	std::initializer_list<uint32_t> il( sizeElements.data(), sizeElements.data() + 4 );
	return create( type, il, shared );
}

eDataType Tensor::getType() const
{
	ID3D11ShaderResourceView* const srv = *this;
	if( nullptr == srv )
		throw OLE_E_BLANK;

	D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
	srv->GetDesc( &viewDesc );
	const DXGI_FORMAT format = viewDesc.Format;
	switch( format )
	{
	case DXGI_FORMAT_R32_FLOAT:
		return eDataType::FP32;
	case DXGI_FORMAT_R16_FLOAT:
		return eDataType::FP16;
	case DXGI_FORMAT_R32_UINT:
		return eDataType::U32;
	}
	throw E_NOTIMPL;
}

CComPtr<ID3D11Buffer> Tensor::getBuffer() const
{
	ID3D11ShaderResourceView* const srv = *this;
	if( nullptr == srv )
		throw OLE_E_BLANK;

	CComPtr<ID3D11Resource> res;
	srv->GetResource( &res );

	CComPtr<ID3D11Buffer> buff;
	check( res.QueryInterface( &buff ) );
	return buff;
}

uint32_t Tensor::dxgiSizeof( DXGI_FORMAT format )
{
	switch( format )
	{
	case DXGI_FORMAT_R16_FLOAT:
		return 2;
	case DXGI_FORMAT_R32_FLOAT:
	case DXGI_FORMAT_R32_UINT:
		return 4;
	}
	throw E_INVALIDARG;
}

void Tensor::downloadImpl( const D3D11_SHADER_RESOURCE_VIEW_DESC& viewDesc, uint32_t countElements, size_t cbElement, void* rdi ) const
{
	assert( viewDesc.ViewDimension == D3D_SRV_DIMENSION_BUFFER );
	const uint32_t idxFirst = viewDesc.Buffer.FirstElement;

	CComPtr<ID3D11Buffer> buff = getBuffer();
	D3D11_BUFFER_DESC desc;
	buff->GetDesc( &desc );
	desc.BindFlags = 0;
	desc.Usage = D3D11_USAGE_STAGING;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

	CComPtr<ID3D11Buffer> staging;
	check( device()->CreateBuffer( &desc, nullptr, &staging ) );
	context()->CopyResource( staging, buff );

	MappedResource mapped;
	check( mapped.map( staging, true ) );
	const uint8_t* rsi = (const uint8_t*)mapped.data();
	rsi += cbElement * idxFirst;
	memcpy( rdi, rsi, cbElement * countElements );
}

void Tensor::download( std::vector<float>& vec ) const
{
	ID3D11ShaderResourceView* const srv = *this;
	if( nullptr == srv )
		throw OLE_E_BLANK;

	D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
	srv->GetDesc( &viewDesc );
	if( viewDesc.Format != DXGI_FORMAT_R32_FLOAT )
		throw E_INVALIDARG;

	uint32_t countElements = viewDesc.Buffer.NumElements;
	vec.resize( countElements );
	downloadImpl( viewDesc, countElements, 4, vec.data() );
}

void Tensor::download( std::vector<uint16_t>& vec ) const
{
	ID3D11ShaderResourceView* const srv = *this;
	if( nullptr == srv )
		throw OLE_E_BLANK;

	D3D11_SHADER_RESOURCE_VIEW_DESC viewDesc;
	srv->GetDesc( &viewDesc );
	if( viewDesc.Format != DXGI_FORMAT_R16_FLOAT )
		throw E_INVALIDARG;

	uint32_t countElements = viewDesc.Buffer.NumElements;
	vec.resize( countElements );
	downloadImpl( viewDesc, countElements, 2, vec.data() );
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