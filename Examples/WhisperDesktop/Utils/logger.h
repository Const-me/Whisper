#pragma once
#include <whisperWindows.h>
#include <cstdarg>

void logMessage( Whisper::eLogLevel lvl, const char8_t* pszFormat, va_list args );

#define LOG_MESSAGE_IMPL( lvl )                \
	std::va_list args;                         \
	va_start( args, pszFormat );               \
	logMessage( lvl, pszFormat, args );        \
	va_end( args )

inline void logError( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( Whisper::eLogLevel::Error );
}
inline void logWarning( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( Whisper::eLogLevel::Warning );
}
inline void logInfo( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( Whisper::eLogLevel::Info );
}
inline void logDebug( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( Whisper::eLogLevel::Debug );
}

#undef LOG_MESSAGE_IMPL

HRESULT logNewSegments( const Whisper::iTranscribeResult* results, size_t newSegments, bool printSpecial = false );

void clearLastError();
bool getLastError( CString& rdi );

void printTime( CStringA& rdi, Whisper::sTimeSpan time, bool comma = false );