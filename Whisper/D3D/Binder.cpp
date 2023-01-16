#include "stdafx.h"
#include "Binder.h"
#include <algorithm>
using namespace DirectCompute;

void Binder::bind( ID3D11ShaderResourceView* srv0, ID3D11UnorderedAccessView* uav0 )
{
	ID3D11DeviceContext* const ctx = context();
	ctx->CSSetUnorderedAccessViews( 0, 1, &uav0, nullptr );

	ctx->CSSetShaderResources( 0, 1, &srv0 );

	maxSrv = std::max( maxSrv, (uint8_t)1 );
	maxUav = std::max( maxUav, (uint8_t)1 );
}

void Binder::bind( ID3D11UnorderedAccessView* uav0 )
{
	context()->CSSetUnorderedAccessViews( 0, 1, &uav0, nullptr );
	maxUav = std::max( maxUav, (uint8_t)1 );
}

void Binder::bind( ID3D11ShaderResourceView* srv0, ID3D11ShaderResourceView* srv1, ID3D11UnorderedAccessView* uav0 )
{
	ID3D11DeviceContext* const ctx = context();
	ctx->CSSetUnorderedAccessViews( 0, 1, &uav0, nullptr );

	std::array< ID3D11ShaderResourceView*, 2> arr = { srv0, srv1 };
	ctx->CSSetShaderResources( 0, 2, arr.data() );

	maxSrv = std::max( maxSrv, (uint8_t)2 );
	maxUav = std::max( maxUav, (uint8_t)1 );
}

void Binder::bind( std::initializer_list<ID3D11ShaderResourceView*> srvs, std::initializer_list<ID3D11UnorderedAccessView*> uavs )
{
	ID3D11DeviceContext* const ctx = context();

	const size_t lengthResources = srvs.size();
	const size_t lengthUnordered = uavs.size();
	assert( lengthResources > 0 && lengthResources < D3D11_COMMONSHADER_INPUT_RESOURCE_REGISTER_COUNT );
	assert( lengthUnordered > 0 && lengthUnordered < D3D11_PS_CS_UAV_REGISTER_COUNT );

	ctx->CSSetUnorderedAccessViews( 0, (UINT)lengthUnordered, uavs.begin(), nullptr );
	ctx->CSSetShaderResources( 0, (UINT)lengthResources, srvs.begin() );

	maxSrv = std::max( maxSrv, (uint8_t)lengthResources );
	maxUav = std::max( maxUav, (uint8_t)lengthUnordered );
}

Binder::~Binder()
{
	uint8_t count = std::max( maxSrv, maxUav );
	if( 0 == count )
		return;
#pragma warning (disable: 6255) // Compiler doesn't know we have very few of these things
	size_t* arr = (size_t*)_alloca( count * sizeof( size_t ) );
	memset( arr, 0, count * sizeof( size_t ) );

	ID3D11DeviceContext* const ctx = context();
	ctx->CSSetShaderResources( 0, maxSrv, (ID3D11ShaderResourceView**)arr );
	ctx->CSSetUnorderedAccessViews( 0, maxUav, (ID3D11UnorderedAccessView**)arr, nullptr );
}