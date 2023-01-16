#pragma once
#include <whisperWindows.h>
#include <deque>
#include <unordered_set>

class AppState;
class DebugConsole
{
	using eLogLevel = Whisper::eLogLevel;

	struct Entry
	{
		eLogLevel level;
		CStringA message;
		HRESULT print( HANDLE hConsole, CString& tempString ) const;
	};

	CComAutoCriticalSection critSec;
	std::deque<Entry> buffer;
	CString tempString;
	CHandle output;

	inline void logSink( eLogLevel lvl, const char* message );
	static void __stdcall logSinkStatic( void* context, eLogLevel lvl, const char* message );

	static BOOL __stdcall consoleHandlerRoutine( DWORD dwCtrlType );

	static DebugConsole* pGlobalInstance;
	void windowClosed();

	std::unordered_set<CButton*> checkboxes;

	CStringA tempStringA;
	void log( eLogLevel lvl, const char* pszFormat, va_list args );

public:
	HRESULT initialize( eLogLevel level = eLogLevel::Debug );
	~DebugConsole();

	HRESULT show();
	HRESULT hide();
	bool isVisible() const { return output; }

	void addCheckbox( CButton& cb );
	void removeCheckbox( CButton& cb );

	static void logMessage( eLogLevel lvl, const char* pszFormat, va_list args );
};

class ConsoleCheckbox
{
	CButton control;
	DebugConsole* console = nullptr;

public:
	HRESULT initialize( HWND dialog, int idc, AppState& state );
	void click();
	~ConsoleCheckbox()
	{
		if( nullptr != console )
			console->removeCheckbox( control );
	}
	void ensureChecked();
};