#include "stdafx.h"

namespace
{
	wchar_t* formatMessage( HRESULT hr )
	{
		wchar_t* err;
		if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
			NULL,
			hr,
			MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ),
			(LPTSTR)&err,
			0,
			NULL ) )
			return err;
		return nullptr;
	}
}

void printError( HRESULT hr )
{
	const wchar_t* err = formatMessage( hr );
	if( nullptr != err )
	{
		fwprintf( stderr, L"%s\n", err );
		LocalFree( (HLOCAL)err );
	}
	else
		fprintf( stderr, "Error code %i (0x%08X)\n", hr, hr );
}