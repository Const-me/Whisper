#include "stdafx.h"
#include "DbgNanTest.h"
#include "../D3D/MappedResource.h"
using namespace DirectCompute;

HRESULT DbgNanTest::create()
{
	ID3D11Device* const dev = DirectCompute::device();

	CD3D11_BUFFER_DESC desc{ 4, D3D11_BIND_UNORDERED_ACCESS };
	CHECK( dev->CreateBuffer( &desc, nullptr, &bufferDefault ) );

	desc.Usage = D3D11_USAGE_STAGING;
	desc.BindFlags = 0;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	CHECK( dev->CreateBuffer( &desc, nullptr, &bufferStaging ) );

	CD3D11_UNORDERED_ACCESS_VIEW_DESC viewDesc{ D3D11_UAV_DIMENSION_BUFFER, DXGI_FORMAT_R32_UINT, 0, 1 };
	CHECK( dev->CreateUnorderedAccessView( bufferDefault, &viewDesc, &uav ) );

	return S_OK;
}

void DbgNanTest::destroy()
{
	uav = nullptr;
	bufferStaging = nullptr;
	bufferDefault = nullptr;
}

bool DbgNanTest::test() const
{
	context()->CopyResource( bufferStaging, bufferDefault );
	MappedResource mapped;
	check( mapped.map( bufferStaging, true ) );
	const BOOL val = *(const BOOL*)mapped.data();
	return val != 0;
}