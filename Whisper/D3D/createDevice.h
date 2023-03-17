// Low-level functions to create and initialize D3D11 device
#pragma once
#include <string>
#include "sGpuInfo.h"

namespace DirectCompute
{
	HRESULT createDevice( const std::wstring& adapter, ID3D11Device** dev, ID3D11DeviceContext** context );

	HRESULT validateFlags( uint32_t flags );

	HRESULT queryDeviceInfo( sGpuInfo& rdi, ID3D11Device* dev, uint32_t flags );

	// Create another device and context, on the same hardware GPU
	HRESULT cloneDevice( ID3D11Device* source, ID3D11Device** dev, ID3D11DeviceContext** context );
}