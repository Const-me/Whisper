#include "miscUtils.h"
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

std::string utf8( const std::wstring& utf16 )
{
	int count = WideCharToMultiByte( CP_UTF8, 0, utf16.c_str(), (int)utf16.length(), nullptr, 0, nullptr, nullptr );
	std::string str( count, 0 );
	WideCharToMultiByte( CP_UTF8, 0, utf16.c_str(), -1, &str[ 0 ], count, nullptr, nullptr );
	return str;
}

std::wstring utf16( const std::string& u8 )
{
	int count = MultiByteToWideChar( CP_UTF8, 0, u8.c_str(), (int)u8.length(), nullptr, 0 );
	std::wstring str( count, 0 );
	MultiByteToWideChar( CP_UTF8, 0, u8.c_str(), (int)u8.length(), &str[ 0 ], count );
	return str;
}

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

void printError( const char* what, HRESULT hr )
{
	const wchar_t* err = formatMessage( hr );
	if( nullptr != err )
	{
		fwprintf( stderr, L"%S: %s\n", what, err );
		LocalFree( (HLOCAL)err );
	}
	else
		fprintf( stderr, "%s: error code %i (0x%08X)\n", what, hr, hr );
}