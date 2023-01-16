#pragma once
#include "../AppState.h"

// Dropdown list which implements language selector control
class LanguageDropdown
{
	HWND m_hWnd = nullptr;
	std::vector<uint32_t> keys;
	int getInitialSelection( AppState& state ) const;

public:
	operator HWND() const
	{
		return m_hWnd;
	}

	// Query language list form the native library, populate the combo box
	// Then load the last saved language selection from registry, and preselect an item.
	void initialize( HWND owner, int idc, AppState& state );

	// Get the ID of the currently selected language, or UINT_MAX if nothing's selected
	uint32_t selectedLanguage();

	// Get the ID of the currently selected language, and store in registry
	void saveSelection( AppState& state );
};