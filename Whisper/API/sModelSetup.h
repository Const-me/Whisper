#pragma once
#include <stdint.h>

namespace Whisper
{
	enum struct eModelImplementation : uint32_t;

	struct sModelSetup
	{
		eModelImplementation impl;
		uint32_t flags;
		const wchar_t* adapter;
	};

	using pfnListAdapters = void( * )( const wchar_t* name, void* pv );
}