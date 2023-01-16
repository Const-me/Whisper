#include "stdafx.h"
#include "Logger.h"
#include "../API/iContext.cl.h"
#include <cstdarg>
#include <atlstr.h>

namespace
{
	wchar_t* formatMessage( HRESULT hr )
	{
		wchar_t* err;
		if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
			NULL,
			hr,
			MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ),
			(LPTSTR)&err,
			0,
			nullptr ) )
			return err;
		return nullptr;
	}

	class Utf
	{
		CStringA utf8;
		CStringW utf16;

		void appendError( HRESULT hr )
		{
			const wchar_t* err = formatMessage( hr );
			if( nullptr != err )
			{
				utf16 += err;
				LocalFree( (HLOCAL)err );
				utf16.TrimRight();
			}
			else
				utf16.AppendFormat( L"error code %i (0x%08X)", hr, hr );
		}

	public:
		const char* print( const char* pszFormat, std::va_list va )
		{
			utf8.FormatV( pszFormat, va );
			return utf8;
		}
		const wchar_t* print( const wchar_t* pszFormat, std::va_list va )
		{
			utf16.FormatV( pszFormat, va );
			return utf16;
		}
		const wchar_t* upcast( const char* message, int len )
		{
			int count = MultiByteToWideChar( CP_UTF8, 0, message, len, nullptr, 0 );
			if( count == 0 )
				return nullptr;
			wchar_t* b = utf16.GetBufferSetLength( len + 1 );
			count = MultiByteToWideChar( CP_UTF8, 0, message, len, b, len );
			utf16.ReleaseBuffer( count );
			return utf16;
		}
		int utf8Length() const
		{
			return utf8.GetLength();
		}
		const wchar_t* printError( HRESULT hr, const char* pszFormat, std::va_list va )
		{
			print( pszFormat, va );
			upcast( utf8, utf8.GetLength() );
			utf16 += L": ";
			appendError( hr );
			return utf16;
		}
		const char* downcast()
		{
			int count = WideCharToMultiByte( CP_UTF8, 0, utf16, utf16.GetLength(), nullptr, 0, nullptr, nullptr );
			char* s = utf8.GetBufferSetLength( count + 1 );
			count = WideCharToMultiByte( CP_UTF8, 0, utf16, utf16.GetLength(), s, count, nullptr, nullptr );
			utf8.ReleaseBufferSetLength( count );
			return utf8;
		}
	};
	thread_local Utf ts_utf;
	using Whisper::eLoggerFlags;

	class Logger : Whisper::sLoggerSetup
	{
		inline bool hasFlag( eLoggerFlags bit ) const
		{
			return 0 != ( (uint8_t)flags & (uint8_t)bit );
		}

		bool useStdError() const
		{
			return hasFlag( eLoggerFlags::UseStandardError );
		}

		static void writeStdError( Whisper::eLogLevel lvl, const char* message, int len )
		{
			const wchar_t* w = ts_utf.upcast( message, len );
			if( nullptr != w )
				fwprintf( stderr, L"%s\n", w );
		}

	public:
		Logger()
		{
			memset( this, 0, sizeof( Logger ) );
		}

		bool willLog( Whisper::eLogLevel lvl ) const
		{
			if( (uint8_t)lvl > (uint8_t)level )
				return false;
			if( useStdError() )
				return true;
			return nullptr != sink;
		}

		void message( Whisper::eLogLevel lvl, const char8_t* pszFormat, std::va_list va ) const
		{
			const char* s = ts_utf.print( (const char*)pszFormat, va );
			auto pfn = sink;
			if( nullptr != pfn )
				pfn( context, lvl, s );
			if( useStdError() )
				writeStdError( lvl, s, ts_utf.utf8Length() );
		}
		void message( Whisper::eLogLevel lvl, const wchar_t* pszFormat, std::va_list va ) const
		{
			Utf& u = ts_utf;
			const wchar_t* w = u.print( pszFormat, va );
			auto pfn = sink;
			if( nullptr != pfn )
				pfn( context, lvl, u.downcast() );
			if( useStdError() )
				fwprintf( stderr, L"%s\n", w );
		}
		void message( Whisper::eLogLevel lvl, HRESULT hr, const char* pszFormat, std::va_list va ) const
		{
			if( hasFlag( eLoggerFlags::SkipFormatMessage ) )
			{
				message( lvl, (const char8_t*)pszFormat, va );
				return;
			}
			Utf& u = ts_utf;
			const wchar_t* w = ts_utf.printError( hr, (const char*)pszFormat, va );
			auto pfn = sink;
			if( nullptr != pfn )
				pfn( context, lvl, u.downcast() );
			if( useStdError() )
				fwprintf( stderr, L"%s\n", w );
		}

		void operator=( const sLoggerSetup& rsi )
		{
			sink = rsi.sink;
			context = rsi.context;
			level = rsi.level;
			flags = rsi.flags;
		}
	};

	static Logger s_logger;
}

bool willLogMessage( Whisper::eLogLevel lvl )
{
	return s_logger.willLog( lvl );
}

using Whisper::eLogLevel;

#define LOG_MESSAGE_IMPL( lvl )                \
	if( !s_logger.willLog( lvl ) )             \
		return;                                \
	std::va_list args;                         \
	va_start( args, pszFormat );               \
	s_logger.message( lvl, pszFormat, args );  \
	va_end( args );

void logError( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Error );
}
void logError16( const wchar_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Error );
}
void logWarning( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Warning );
}
void logWarning16( const wchar_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Warning );
}
void logInfo( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Info );
}
void logInfo16( const wchar_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Info );
}
void logDebug( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Debug );
}
void logDebug16( const wchar_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Debug );
}
#undef LOG_MESSAGE_IMPL

#define LOG_MESSAGE_IMPL( lvl )                \
	if( !s_logger.willLog( lvl ) )             \
		return;                                \
	std::va_list args;                         \
	va_start( args, pszFormat );               \
	s_logger.message( lvl, hr, (const char*)pszFormat, args );  \
	va_end( args );

void logErrorHr( long hr, const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Error );
}
void logWarningHr( long hr, const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( eLogLevel::Warning );
}

#undef LOG_MESSAGE_IMPL

// DLL entry point
HRESULT COMLIGHTCALL Whisper::setupLogger( const sLoggerSetup& setup )
{
	s_logger = setup;
	return S_OK;
}