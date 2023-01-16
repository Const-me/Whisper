#include "stdafx.h"
#include "CircleIndicator.h"
#include <atltypes.h>
#include "AppState.h"

namespace
{
	static const LPCTSTR windowClassName = L"CircleIndicator";

	// Font with these symbols, shipped with Windows since forever:
	// https://learn.microsoft.com/en-us/typography/font-list/segoe-ui-symbol
	static const LPCTSTR fontName = L"Segoe UI Symbol";

	// Outlined circle
	static const LPCTSTR circleOutline = L"⚪";
	// Filled circle
	static const LPCTSTR circleFilled = L"⚫";

	// Font size for that symbol font, in points
	constexpr int fontSizePoints = 14;

	// Default active color for the indicator
	constexpr uint32_t defaultActiveColor = 0x7FFF7F;
}

CircleIndicator::CircleIndicator() :
	activeColor( defaultActiveColor )
{ }

ATL::CWndClassInfo& CircleIndicator::GetWndClassInfo()
{
	// Use custom class style with CS_PARENTDC, and COLOR_3DFACE for the background
	static ATL::CWndClassInfo wc = 
	{
		{ sizeof( WNDCLASSEX ),
		CS_HREDRAW | CS_VREDRAW | CS_PARENTDC,
		StartWindowProc,
		0, 0, NULL, NULL, NULL, (HBRUSH)( COLOR_3DFACE + 1 ), NULL, windowClassName, NULL },
		NULL, NULL, IDC_ARROW, TRUE, 0, _T( "" )
	};
	return wc;
}

// Class registration
HRESULT CircleIndicator::registerClass()
{
	WNDPROC pUnusedWndSuperProc = nullptr;
	ATOM a = GetWndClassInfo().Register( &pUnusedWndSuperProc );
	if( 0 != a )
		return S_OK;
	return getLastHr();
}

HRESULT CircleIndicator::createFont( int height )
{
	LOGFONT logFont;
	memset( &logFont, 0, sizeof( logFont ) );
	logFont.lfHeight = height;
	logFont.lfCharSet = ANSI_CHARSET;
	logFont.lfOutPrecision = OUT_TT_PRECIS;
	logFont.lfClipPrecision = CLIP_DEFAULT_PRECIS;
	wcsncpy_s( logFont.lfFaceName, fontName, _TRUNCATE );
	font.CreateFontIndirect( &logFont );
	if( font )
		return S_OK;
	return E_FAIL;
}

void CircleIndicator::onDestroy()
{
	if( font )
		font.DeleteObject();
}

void CircleIndicator::onPaint( CDCHandle dc )
{
	CRect rectInt32;
	GetClientRect( &rectInt32 );

	CPaintDC pdc( m_hWnd );

	const int logPixels = pdc.GetDeviceCaps( LOGPIXELSY );
	int fontSize = -MulDiv( fontSizePoints, logPixels, 72 );
	if( !font || fontHeight != fontSize )
	{
		if( font )
			font.DeleteObject();
		HRESULT hr = createFont( fontSize );
		if( FAILED( hr ) )
			return;
		fontHeight = fontSize;
	}

	pdc.SetBkColor( GetSysColor( COLOR_3DFACE ) );
	pdc.SelectFont( font );
	pdc.SetBkMode( OPAQUE );
	constexpr UINT textFormat = DT_CENTER | DT_VCENTER;

	if( isActive )
	{
		pdc.SetTextColor( activeColor );
		pdc.DrawText( circleFilled, 1, rectInt32, textFormat );
		pdc.SetBkMode( TRANSPARENT );
	}

	pdc.SetTextColor( 0 );
	pdc.DrawText( circleOutline, 1, rectInt32, textFormat );
}

void CircleIndicator::setActive( bool nowActive )
{
	if( nowActive == isActive )
		return;

	// Repaint the control
	isActive = nowActive;
	InvalidateRect( nullptr );
}