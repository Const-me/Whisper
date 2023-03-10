#pragma once
#include <string>
using HRESULT = long;

namespace DirectCompute
{
	HRESULT d3dStartup( uint32_t flags, const std::wstring& adapter );
	void d3dShutdown();

	HRESULT createComputeShaders();
	void destroyComputeShaders();
}