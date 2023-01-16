#include "stdafx.h"
#include "TensorGpuViews.h"
using namespace DirectCompute;

HRESULT TensorGpuViews::create( ID3D11Buffer* gpuBuffer, DXGI_FORMAT format, size_t countElements, bool makeUav )
{
	srv = nullptr;
	uav = nullptr;

	if( countElements > UINT_MAX )
		return DISP_E_OVERFLOW;

	CD3D11_SHADER_RESOURCE_VIEW_DESC viewDesc{ D3D11_SRV_DIMENSION_BUFFER, format, 0, (UINT)countElements };
	CHECK( device()->CreateShaderResourceView( gpuBuffer, &viewDesc, &srv ) );

	if( makeUav )
	{
		CD3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{ D3D11_UAV_DIMENSION_BUFFER, format , 0, (UINT)countElements };
		CHECK( device()->CreateUnorderedAccessView( gpuBuffer, &uavDesc, &uav ) );
	}

	return S_OK;
}