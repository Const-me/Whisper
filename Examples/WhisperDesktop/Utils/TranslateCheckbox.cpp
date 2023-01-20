#include "stdafx.h"
#include "TranslateCheckbox.h"

static const LPCTSTR regValTranslate = L"translate";

void TranslateCheckbox::initialize( HWND owner, int idc, AppState& state )
{
	m_hWnd = GetDlgItem( owner, idc );
	assert( nullptr != m_hWnd );

	if( state.boolLoad( regValTranslate ) )
		::SendMessage( m_hWnd, BM_SETCHECK, BST_CHECKED, 0L );
}

bool TranslateCheckbox::checked()
{
	assert( nullptr != m_hWnd );
	const int state = ( int )::SendMessage( m_hWnd, BM_GETCHECK, 0, 0 );
	return state == BST_CHECKED;
}

void TranslateCheckbox::saveSelection( AppState& state )
{
	state.boolStore( regValTranslate, checked() );
}