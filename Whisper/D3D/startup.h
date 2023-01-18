#pragma once
using HRESULT = long;

namespace DirectCompute
{
	HRESULT d3dStartup( uint32_t flags );
	void d3dShutdown();

	HRESULT createComputeShaders();
	void destroyComputeShaders();
}