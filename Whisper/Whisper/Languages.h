#pragma once
#include "../../ComLightLib/comLightCommon.h"

namespace Whisper
{
	int lookupLanguageId( const char* code );
	int lookupLanguageId( uint32_t key );

	const char* lookupLanguageName( const char* code );

	int COMLIGHTCALL getLanguageId( const char* lang );
}