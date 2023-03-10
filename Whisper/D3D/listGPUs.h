#pragma once
#include <API/iContext.cl.h>

namespace DirectCompute
{
	CComPtr<IDXGIAdapter1> selectAdapter( const std::wstring& requestedName );
}