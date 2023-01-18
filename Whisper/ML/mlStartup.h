#pragma once
using HRESULT = long;

namespace DirectCompute
{
	HRESULT mlStartup( uint32_t flags );
	void mlShutdown();
}