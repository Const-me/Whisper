#pragma once
#include <iContext.h>
#include "logger.h"

CString formatErrorMessage( HRESULT hr );

void reportFatalError( const char* what, HRESULT hr );

#define CHECK( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) return __hr; }

HRESULT implParse( const CString& s, Whisper::eModelImplementation& rdi );

LPCTSTR implString( Whisper::eModelImplementation i );

void implPopulateCombobox( CComboBox& cb, Whisper::eModelImplementation impl );

Whisper::eModelImplementation implGetValue( CComboBox& cb );

__interface iThreadPoolCallback
{
	void __stdcall poolCallback() noexcept;
};

class ThreadPoolWork
{
	PTP_WORK work = nullptr;
	static void __stdcall callback( PTP_CALLBACK_INSTANCE Instance, PVOID Context, PTP_WORK Work );

public:

	~ThreadPoolWork();
	HRESULT create( iThreadPoolCallback* cb );
	HRESULT post();
};

void makeUtf16( CString& rdi, const char* utf8 );
void makeUtf8( CStringA& rdi, const CString& utf16 );

bool getOpenFileName( HWND owner, LPCTSTR title, LPCTSTR filter, CString& path );

bool getSaveFileName( HWND owner, LPCTSTR title, LPCTSTR filter, CString& path, DWORD* filterIndex = nullptr );

#define ON_BUTTON_CLICK( id, func )  \
	if( uMsg == WM_COMMAND &&        \
         id == LOWORD( wParam ) )    \
	{                                \
		bHandled = TRUE;             \
		func();                      \
		lResult = 0;                 \
		return TRUE;                 \
	}

void reportError( HWND owner, LPCTSTR text, LPCTSTR title, HRESULT hr = S_FALSE );

inline const wchar_t* cstr( const CString& s ) { return s; }
inline const char* cstr( const CStringA& s ) { return s; }

inline HRESULT getLastHr()
{
	return HRESULT_FROM_WIN32( GetLastError() );
}

HRESULT writeUtf8Bom( CAtlFile& file );

// Flip order of bytes from RGB to BGR, or vice versa
inline uint32_t flipRgb( uint32_t val )
{
	val = _byteswap_ulong( val );
	return val >> 8;
}

bool isInvalidTranslate( HWND owner, uint32_t lang, bool translate );

inline bool isChecked( CButton& btn )
{
	return btn.GetCheck() == BST_CHECKED;
}