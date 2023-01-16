#pragma once
#include "../API/loggerApi.h"

#ifdef  __cplusplus
extern "C" {
#endif

void logError( const char8_t* pszFormat, ... );
void logError16( const wchar_t* pszFormat, ... );
void logErrorHr( long hr, const char8_t* pszFormat, ... );
void logWarning( const char8_t* pszFormat, ... );
void logWarning16( const wchar_t* pszFormat, ... );
void logWarningHr( long hr, const char8_t* pszFormat, ... );
void logInfo( const char8_t* pszFormat, ... );
void logInfo16( const wchar_t* pszFormat, ... );
void logDebug( const char8_t* pszFormat, ... );
void logDebug16( const wchar_t* pszFormat, ... );

bool willLogMessage( Whisper::eLogLevel lvl );

#ifdef  __cplusplus
}
#endif
