#include "stdafx.h"
#include "Languages.h"
#include <atlcoll.h>
#include "../API/iContext.cl.h"

namespace
{
	// These structures are compiled into the DLL, in read only data section
	using Lang = Whisper::sLanguageEntry;

	static const Lang s_languageData[] =
	{
#include "languageCodez.inl"
	};

	using Whisper::makeLanguageKey;

	// Values for the hash map
	struct sLanguage
	{
		int id;
		const char* name;
	};

	class LanguageIDs
	{
		CAtlMap<uint32_t, sLanguage> map;

		void add( const char* code, int id, const char* name )
		{
			assert( strlen( code ) <= 4 );
			const uint16_t key = makeLanguageKey( code );
			map.SetAt( key, sLanguage{ id, name } );
		}

	public:

		LanguageIDs() :
			map( 103u, 0.75f, 0.25f, 2.25f, 99 )
		{
			for( const Lang& e : s_languageData )
				map.SetAt( e.key, sLanguage{ e.id, e.name } );
		};

		int lookupId( const char* code ) const
		{
			const uint32_t key = makeLanguageKey( code );
			auto p = map.Lookup( key );
			return ( nullptr != p ) ? p->m_value.id : -1;
		}

		int lookupKey( uint32_t key ) const
		{
			auto p = map.Lookup( key );
			return ( nullptr != p ) ? p->m_value.id : -1;
		}

		const char* lookupName( const char* code ) const
		{
			const uint32_t key = makeLanguageKey( code );
			auto p = map.Lookup( key );
			return ( nullptr != p ) ? p->m_value.name : nullptr;
		}
	};

	static const LanguageIDs g_table;
}

namespace Whisper
{
	int lookupLanguageId( const char* code )
	{
		return g_table.lookupId( code );
	}
	int lookupLanguageId( uint32_t key )
	{
		return g_table.lookupKey( key );
	}
	const char* lookupLanguageName( const char* code )
	{
		return g_table.lookupName( code );
	}
	int COMLIGHTCALL getLanguageId( const char* lang )
	{
		return lookupLanguageId( lang );
	}

	uint32_t COMLIGHTCALL findLanguageKeyW( const wchar_t* lang )
	{
		uint32_t key = 0;
		uint32_t shift = 0;
		for( size_t i = 0; i < 4; i++, lang++, shift += 8 )
		{
			const wchar_t c = *lang;
			if( c == L'\0' )
				break;
			if( c >= 0x80 )
				return UINT_MAX;
			uint32_t u32 = (uint8_t)c;
			u32 = u32 << shift;
			key |= u32;
		}
		if( g_table.lookupKey( key ) >= 0 )
			return key;
		return UINT_MAX;
	}

	uint32_t COMLIGHTCALL findLanguageKeyA( const char* lang )
	{
		const uint32_t key = makeLanguageKey( lang );
		if( g_table.lookupKey( key ) >= 0 )
			return key;
		return UINT_MAX;
	}

	HRESULT COMLIGHTCALL getSupportedLanguages( sLanguageList& rdi )
	{
		rdi.length = sizeof( s_languageData ) / sizeof( s_languageData[ 0 ] );
		rdi.pointer = s_languageData;
		return S_OK;
	}
}