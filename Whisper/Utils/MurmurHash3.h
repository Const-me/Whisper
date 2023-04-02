#pragma once
#include <stdint.h>

void MurmurHash3_x86_32( const void* key, int len, uint32_t seed, void* out );
void MurmurHash3_x86_128( const void* key, int len, uint32_t seed, void* out );
void MurmurHash3_x64_128( const void* key, int len, uint32_t seed, void* out );

#include <atlcoll.h>

// Traits class for `CAtlMap<const char*>` which does not copy nor owns these strings
struct StringPtrTraits : public ATL::CDefaultElementTraits<const char*>
{
	using INARGTYPE = const char*;

	static inline bool CompareElements( const char* a, const char* b )
	{
		return 0 == strcmp( a, b );
	}

	static inline int CompareElementsOrdered( const char* a, const char* b )
	{
		return strcmp( a, b );
	}

	static inline ULONG Hash( const char* ptr )
	{
		uint32_t hash = UINT_MAX;
		if( nullptr != ptr )
		{
			const int len = (int)strlen( ptr );
			constexpr uint32_t seed = 0;
			MurmurHash3_x86_32( ptr, len, seed, &hash );
		}
		return hash;
	}
};