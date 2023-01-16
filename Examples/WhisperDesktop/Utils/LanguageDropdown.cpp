#include "stdafx.h"
#include "LanguageDropdown.h"
#include "miscUtils.h"

namespace
{
	inline wchar_t toUpper( wchar_t c )
	{
		size_t st = (uint16_t)c;
		st = reinterpret_cast<size_t>( CharUpperW( reinterpret_cast<LPWSTR>( st ) ) );
		return (wchar_t)(uint16_t)st;
	}

	void makeTitleCase( CString& s )
	{
		bool cap = true;
		for( int i = 0; i < s.GetLength(); i++ )
		{
			wchar_t c = s[ i ];
			if( cap )
			{
				c = toUpper( c );
				s.SetAt( i, c );
			}
			cap = false;
			if( c == ' ' )
				cap = true;
		}
	}
}

int LanguageDropdown::getInitialSelection( AppState& state ) const
{
	constexpr uint32_t english = 0x6E65;

	// Load preference from the registry
	uint32_t id = state.languageRead();
	if( id == UINT_MAX )
		id = english;

	auto it = std::find( keys.begin(), keys.end(), id );
	if( it == keys.end() )
	{
		id = english;
		it = std::find( keys.begin(), keys.end(), id );
		assert( it != keys.end() );
	}

	ptrdiff_t idx = it - keys.begin();
	return (int)idx;
}

void LanguageDropdown::initialize( HWND owner, int idc, AppState& state )
{
	m_hWnd = GetDlgItem( owner, idc );
	assert( nullptr != m_hWnd );

	Whisper::sLanguageList list;
	Whisper::getSupportedLanguages( list );

	const size_t length = list.length;
	keys.resize( length );
	CString utf16;
	for( size_t i = 0; i < length; i++ )
	{
		keys[ i ] = list.pointer[ i ].key;
		makeUtf16( utf16, list.pointer[ i ].name );
		makeTitleCase( utf16 );
		SendMessage( m_hWnd, CB_ADDSTRING, 0, (LPARAM)cstr( utf16 ) );
	}

	const int curSel = getInitialSelection( state );
	SendMessage( m_hWnd, CB_SETCURSEL, curSel, 0 );
}

uint32_t LanguageDropdown::selectedLanguage()
{
	const int cs = (int)SendMessage( m_hWnd, CB_GETCURSEL, 0, 0 );
	if( cs < 0 || cs >= keys.size() )
		return UINT_MAX;
	return keys[ cs ];
}

void LanguageDropdown::saveSelection( AppState& state )
{
	state.languageWrite( selectedLanguage() );
}