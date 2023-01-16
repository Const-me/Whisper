#pragma once
using HRESULT = long;

namespace DirectCompute
{
	HRESULT d3dStartup();
	void d3dShutdown();

	HRESULT createComputeShaders();
	void destroyComputeShaders();
}