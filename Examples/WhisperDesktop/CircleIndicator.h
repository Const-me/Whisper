#pragma once
#include "Utils/miscUtils.h"
#include "Utils/WTL/atlcrack.h"

// This control renders a black circle, and in the active state, the circle is filled with a bright color.
class CircleIndicator: public CWindowImpl<CircleIndicator>
{
public:
	static ATL::CWndClassInfo& GetWndClassInfo();

	BEGIN_MSG_MAP( CircleIndicator )
		MSG_WM_PAINT( onPaint )
		MSG_WM_DESTROY( onDestroy )
	END_MSG_MAP()

	// Class registration
	static HRESULT registerClass();

	void setActive( bool nowActive );

	void setActiveColor( uint32_t col )
	{
		activeColor = col;
	}
	CircleIndicator();

private:
	bool isActive = false;
	uint32_t activeColor;
	int fontHeight = 0;
	CFont font;
	HRESULT createFont( int height );

	void onDestroy();
	void onPaint( CDCHandle dc );
};