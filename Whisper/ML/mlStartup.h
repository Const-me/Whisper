#pragma once
#include <string>
using HRESULT = long;

namespace DirectCompute
{
	HRESULT mlStartup( uint32_t flags, const std::wstring& adapter );
	void mlShutdown();
}