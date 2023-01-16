#pragma once

// Utility class to switch visual state of dialog controls between idle and pending
class PendingState
{
	std::vector<HWND> editorsWindows;
	std::vector<bool> wasEnabled;
	std::vector<HWND> pendingWindows;
public:
	void initialize( std::initializer_list<HWND> editors, std::initializer_list<HWND> pending );
	void setPending( bool nowPending );
};