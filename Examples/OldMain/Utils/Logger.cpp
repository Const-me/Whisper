#include <stdint.h>
#include <vector>
#include <cstdarg>
#include "Logger.h"

namespace
{
	void logMessage( const char* lvl, const char8_t* pszFormat, std::va_list va )
	{
		fprintf( stderr, "%s: ", lvl );
		vfprintf( stderr, (const char*)pszFormat, va );
		fprintf( stderr, "\n" );
	}
}

#define LOG_MESSAGE_IMPL( lvl )                \
	std::va_list args;                         \
	va_start( args, pszFormat );               \
	logMessage( lvl, pszFormat, args );        \
	va_end( args );

void logError( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( "Error" );
}

void logWarning( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( "Warning" );
}

void logInfo( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( "Info" );
}

void logDebug( const char8_t* pszFormat, ... )
{
	LOG_MESSAGE_IMPL( "Debug" );
}