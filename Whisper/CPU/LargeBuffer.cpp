#include "stdafx.h"
#include "LargeBuffer.h"
using namespace CpuCompute;

void LargeBuffer::deallocate()
{
	if( nullptr == pv )
		return;
	VirtualFree( pv, 0, MEM_RELEASE );
	pv = nullptr;
}

HRESULT LargeBuffer::allocate( size_t cb )
{
	deallocate();
	
	pv = VirtualAlloc( nullptr, cb, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE );
	if( nullptr != pv )
		return S_OK;
	return HRESULT_FROM_WIN32( GetLastError() );
}

HRESULT LargeBuffer::setReadOnly( size_t cb )
{
	if( nullptr != pv )
	{
		DWORD op = 0;
		if( VirtualProtect( pv, cb, PAGE_READONLY, &op ) )
			return S_OK;
		return HRESULT_FROM_WIN32( GetLastError() );
	}
	else
		return OLE_E_BLANK;
}