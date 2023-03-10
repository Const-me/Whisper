// https://github.com/Const-me/vis_avs_dx/blob/master/avs_dx/DxVisuals/Interop/ConsoleLogger.cpp
#include "stdafx.h"
#include "DebugConsole.h"
#include "miscUtils.h"
#include "../AppState.h"
#include "logger.h"
#include <string>

namespace
{
	using Whisper::eLogLevel;

	constexpr uint16_t defaultAttributes = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;

	inline uint16_t textAttributes( eLogLevel lvl )
	{
		switch( lvl )
		{
		case eLogLevel::Error:
			return FOREGROUND_RED | FOREGROUND_INTENSITY;
		case eLogLevel::Warning:
			return FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
		case eLogLevel::Info:
			return FOREGROUND_GREEN | FOREGROUND_INTENSITY;
		case eLogLevel::Debug:
			return FOREGROUND_BLUE | FOREGROUND_INTENSITY;
		}
		return defaultAttributes;
	}

	// Background stuff: accumulate messages in a small buffer, in case user will want to see them in the console.
	// Ideally, we should accumulate them in a more efficient data structure, maybe a circular buffer.
	// However, we don't have that many messages per second, this simple solution that uses std::deque is probably good enough for the job.
	static constexpr uint16_t bufferSize = 64;

	using Lock = CComCritSecLock<CComAutoCriticalSection>;
#define LOCK() Lock __lock{ critSec }

	thread_local std::string threadError;
}

HRESULT DebugConsole::Entry::print( HANDLE hConsole, CString& tempString ) const
{
	if( !SetConsoleTextAttribute( hConsole, textAttributes( level ) ) )
		return getLastHr();

	makeUtf16( tempString, message );
	tempString += L"\r\n";
	if( !WriteConsoleW( hConsole, tempString, (DWORD)tempString.GetLength(), nullptr, nullptr ) )
		return getLastHr();
	return S_OK;
}

void clearLastError()
{
	threadError.clear();
}

bool getLastError( CString& rdi )
{
	if( threadError.empty() )
	{
		rdi = L"";
		return false;
	}
	else
	{
		makeUtf16( rdi, threadError.c_str() );
		threadError.clear();
		return true;
	}
}

inline void DebugConsole::logSink( eLogLevel lvl, const char* message )
{
	LOCK();

	// Add to the buffer
	while( buffer.size() >= bufferSize )
		buffer.pop_front();
	buffer.emplace_back( Entry{ lvl, message } );

	// If the console window is shown, print there, too.
	if( output )
		buffer.rbegin()->print( output, tempString );
}

void __stdcall DebugConsole::logSinkStatic( void* context, eLogLevel lvl, const char* message )
{
	if( lvl == eLogLevel::Error )
		threadError = message;

	DebugConsole* con = (DebugConsole*)context;
	con->logSink( lvl, message );
}

HRESULT DebugConsole::initialize( Whisper::eLogLevel level )
{
	if( nullptr != pGlobalInstance )
		return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );
	pGlobalInstance = this;

	Whisper::sLoggerSetup setup;
	setup.sink = &logSinkStatic;
	setup.context = this;
	setup.level = level;
	setup.flags = Whisper::eLoggerFlags::SkipFormatMessage;
	return Whisper::setupLogger( setup );
}

DebugConsole::~DebugConsole()
{
	hide();

	Whisper::sLoggerSetup setup;
	memset( &setup, 0, sizeof( setup ) );
	Whisper::setupLogger( setup );

	pGlobalInstance = nullptr;
}

DebugConsole* DebugConsole::pGlobalInstance = nullptr;

void DebugConsole::windowClosed()
{
	LOCK();
	if( FreeConsole() )
	{
		// Apparently, FreeConsole already closes that handle: https://stackoverflow.com/q/12676312/126995
		output.Detach();
	}
	output.Close();

	for( CButton* b : checkboxes )
	{
		if( !*b )
			continue;
		if( !b->IsWindow() )
			continue;
		PostMessage( *b, BM_SETCHECK, BST_UNCHECKED, 0 );
	}
}

BOOL __stdcall DebugConsole::consoleHandlerRoutine( DWORD dwCtrlType )
{
	switch( dwCtrlType )
	{
	case CTRL_CLOSE_EVENT:
	case CTRL_C_EVENT:
	case CTRL_BREAK_EVENT:
		pGlobalInstance->windowClosed();
		return TRUE;
	}
	return TRUE;
}

HRESULT DebugConsole::show()
{
	HWND hWnd = GetConsoleWindow();
	if( nullptr != hWnd )
	{
		ShowWindow( hWnd, SW_RESTORE );
		SetForegroundWindow( hWnd );
		return S_FALSE;
	}

	if( !AllocConsole() )
		return getLastHr();

	output.Close();
	output.Attach( GetStdHandle( STD_OUTPUT_HANDLE ) );
	if( !output )
		return getLastHr();

	constexpr UINT cp = CP_UTF8;
	if( IsValidCodePage( cp ) )
		SetConsoleOutputCP( cp );

	// Enable ANSI color coding
	DWORD mode = 0;
	if( !GetConsoleMode( output, &mode ) )
		return getLastHr();
	if( 0 == ( mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING ) )
	{
		mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
		if( !SetConsoleMode( output, mode ) )
			return getLastHr();
	}

	SetConsoleTitle( L"Whisper Desktop Debug Console" );

	SetConsoleCtrlHandler( &consoleHandlerRoutine, TRUE );

	// Disable close command in the sys.menu of the new console, otherwise the whole process will quit: https://stackoverflow.com/a/12015131/126995
	HWND hwnd = ::GetConsoleWindow();
	if( hwnd != nullptr )
	{
		HMENU hMenu = ::GetSystemMenu( hwnd, FALSE );
		if( hMenu != NULL )
			DeleteMenu( hMenu, SC_CLOSE, MF_BYCOMMAND );
	}

	// Print old log entries
	for( const auto& e : buffer )
		CHECK( e.print( output, tempString ) );

	const CStringA msg = "Press Control+C or Control+Break to close this window\r\n";
	if( !SetConsoleTextAttribute( output, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY ) )
		return getLastHr();
	if( !WriteConsoleA( output, cstr( msg ), msg.GetLength(), nullptr, nullptr ) )
		return getLastHr();

	return S_OK;
}

HRESULT DebugConsole::hide()
{
	LOCK();
	if( !output )
		return S_FALSE;
	windowClosed();
	return S_OK;
}

void DebugConsole::addCheckbox( CButton& cb )
{
	checkboxes.emplace( &cb );
}
void DebugConsole::removeCheckbox( CButton& cb )
{
	checkboxes.erase( &cb );
}

HRESULT ConsoleCheckbox::initialize( HWND dialog, int idc, AppState& state )
{
	control = GetDlgItem( dialog, idc );
	assert( control );

	console = &state.console;
	if( state.console.isVisible() )
		control.SetCheck( BST_CHECKED );

	state.console.addCheckbox( control );
	return S_OK;
}

void ConsoleCheckbox::click()
{
	const int state = control.GetCheck();
	if( state == BST_CHECKED )
		console->show();
	else
		console->hide();
}

void ConsoleCheckbox::ensureChecked()
{
	const int state = control.GetCheck();
	if( state == BST_CHECKED )
		return;
	control.SetCheck( BST_CHECKED );
	console->show();
}

void DebugConsole::log( eLogLevel lvl, const char* pszFormat, va_list args )
{
	LOCK();
	// Add to the buffer
	while( buffer.size() >= bufferSize )
		buffer.pop_front();

	tempStringA.FormatV( pszFormat, args );
	buffer.emplace_back( Entry{ lvl, tempStringA } );

	// If the console window is shown, print there, too.
	if( output )
		buffer.rbegin()->print( output, tempString );
}

void DebugConsole::logMessage( eLogLevel lvl, const char* pszFormat, va_list args )
{
	if( nullptr == pGlobalInstance )
		return;
	pGlobalInstance->log( lvl, pszFormat, args );
}

void logMessage( Whisper::eLogLevel lvl, const char8_t* pczFormat, va_list args )
{
	DebugConsole::logMessage( lvl, (const char*)pczFormat, args );
}