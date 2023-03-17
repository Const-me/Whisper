#pragma once
#include <atlcomcli.h>
#include <string>
#include "sGpuInfo.h"

namespace DirectCompute
{
	ID3D11Device* device();
	ID3D11DeviceContext* context();
	D3D_FEATURE_LEVEL featureLevel();

	HRESULT initialize( uint32_t flags, const std::wstring& adapter );
	void terminate();

	const sGpuInfo& gpuInfo();

	inline bool available()
	{
		return nullptr != device();
	}

	inline void csSetCB( ID3D11Buffer* cb )
	{
		context()->CSSetConstantBuffers( 0, 1, &cb );
	}

	__m128i bufferMemoryUsage( ID3D11Buffer* buffer );

	__m128i resourceMemoryUsage( ID3D11ShaderResourceView* srv );
}