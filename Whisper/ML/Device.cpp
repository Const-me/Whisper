#include "stdafx.h"
#include "Device.h"
#include "../D3D/createDevice.h"
#include "../D3D/shaders.h"
#include "../D3D/device.h"
#include "mlUtils.h"
#include "../D3D/MappedResource.h"

using namespace DirectCompute;

HRESULT Device::create( uint32_t flags, const std::wstring& adapter )
{
	CHECK( validateFlags( flags ) );
	CHECK( createDevice( adapter, &device, &context ) );
	CHECK( queryDeviceInfo( gpuInfo, device, flags ) );

	CHECK( createComputeShaders( shaders ) );

	CHECK( lookupTables.create() );
	{
		CD3D11_BUFFER_DESC desc{ 16, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE };
		CHECK( device->CreateBuffer( &desc, nullptr, &smallCb ) );
	}

#if DBG_TEST_NAN
	CHECK( nanTestBuffers.create() );
#endif

	return S_OK;
}

HRESULT Device::createClone( const Device& source )
{
	CHECK( cloneDevice( source.device, &device, &context ) );
	gpuInfo = source.gpuInfo;

	CHECK( createComputeShaders( shaders ) );
	CHECK( lookupTables.createClone( source.lookupTables ) );

	{
		CD3D11_BUFFER_DESC desc{ 16, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE };
		CHECK( device->CreateBuffer( &desc, nullptr, &smallCb ) );
	}

#if DBG_TEST_NAN
	CHECK( nanTestBuffers.create() );
#endif

	return S_OK;
}

void Device::destroy()
{
#if DBG_TEST_NAN
	nanTestBuffers.destroy();
#endif
	smallCb = nullptr;
	shaders.clear();
	context = nullptr;
	device = nullptr;
}

__m128i __declspec( noinline ) DirectCompute::bufferMemoryUsage( ID3D11Buffer* buffer )
{
	if( nullptr != buffer )
	{
		D3D11_BUFFER_DESC desc;
		buffer->GetDesc( &desc );

		if( desc.Usage != D3D11_USAGE_STAGING )
			return setHigh_size( desc.ByteWidth );
		else
			return setLow_size( desc.ByteWidth );
	}
	return _mm_setzero_si128();
}

__m128i __declspec( noinline ) DirectCompute::resourceMemoryUsage( ID3D11ShaderResourceView* srv )
{
	if( nullptr != srv )
	{
		CComPtr<ID3D11Resource> res;
		srv->GetResource( &res );
		CComPtr<ID3D11Buffer> buff;
		if( SUCCEEDED( res.QueryInterface( &buff ) ) )
			return bufferMemoryUsage( buff );
		assert( false );	// We don't use textures in this project
	}
	return _mm_setzero_si128();
}

static thread_local const Device* ts_device = nullptr;

ID3D11Device* DirectCompute::device()
{
	const Device* dev = ts_device;
	if( nullptr != dev )
		return dev->device;
	throw OLE_E_BLANK;
}

ID3D11DeviceContext* DirectCompute::context()
{
	const Device* dev = ts_device;
	if( nullptr != dev )
		return dev->context;
	throw OLE_E_BLANK;
}

const sGpuInfo& DirectCompute::gpuInfo()
{
	const Device* dev = ts_device;
	if( nullptr != dev )
		return dev->gpuInfo;
	throw OLE_E_BLANK;
}

const LookupTables& DirectCompute::lookupTables()
{
	const Device* dev = ts_device;
	if( nullptr != dev )
		return dev->lookupTables;
	throw OLE_E_BLANK;
}

void DirectCompute::bindShader( eComputeShader shader )
{
	const Device* dev = ts_device;
	if( nullptr != dev )
	{
		ID3D11ComputeShader* cs = dev->shaders[ (uint16_t)shader ];
		dev->context->CSSetShader( cs, nullptr, 0 );
		return;
	}
	throw OLE_E_BLANK;
}

ID3D11Buffer* __vectorcall DirectCompute::updateSmallCb( __m128i cbData )
{
	const Device* dev = ts_device;
	if( nullptr != dev && nullptr != dev->smallCb )
	{
		ID3D11Buffer* cb = dev->smallCb;
		MappedResource mapped;
		check( mapped.map( cb, false ) );
		store16( mapped.data(), cbData );
		return cb;
	}

	throw OLE_E_BLANK;
}

#if DBG_TEST_NAN
const DbgNanTest& DirectCompute::getNanTestBuffers()
{
	const Device* dev = ts_device;
	if( nullptr != dev )
		return dev->nanTestBuffers;
	throw OLE_E_BLANK;
}
#endif

Device::ThreadSetupRaii::ThreadSetupRaii( const Device* dev )
{
	assert( ts_device == nullptr );
	ts_device = dev;
	setup = true;
}

Device::ThreadSetupRaii::~ThreadSetupRaii()
{
	if( setup )
	{
		assert( ts_device != nullptr );
		ts_device = nullptr;
	}
}