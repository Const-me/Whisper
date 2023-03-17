#pragma once
#include "enums.h"
#include "device.h"

namespace DirectCompute
{
	HRESULT createBuffer( eBufferUse use, size_t totalBytes, ID3D11Buffer** ppGpuBuffer, const void* rsi, ID3D11Buffer** ppStagingBuffer, bool shared = false );
}