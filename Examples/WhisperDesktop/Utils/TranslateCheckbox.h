#pragma once
#include "../AppState.h"

class TranslateCheckbox
{
	HWND m_hWnd = nullptr;
public:
	operator HWND() const
	{
		return m_hWnd;
	}

	void initialize( HWND owner, int idc, AppState& state );

	bool checked();

	void saveSelection( AppState& state );
};