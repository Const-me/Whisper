#pragma once
using HRESULT = long;

namespace DirectCompute
{
	HRESULT mlStartup();
	void mlShutdown();
}