#pragma once
#include <API/iContext.cl.h>

namespace DirectCompute
{
	void destroyDxgiFactory();

	CComPtr<IDXGIAdapter1> selectAdapter( const std::wstring& requestedName );
}