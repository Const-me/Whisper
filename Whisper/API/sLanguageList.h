#pragma once
#include <stdint.h>

namespace Whisper
{
	struct sLanguageEntry
	{
		uint32_t key;
		int id;
		const char* name;
	};

	struct sLanguageList
	{
		uint32_t length;
		const sLanguageEntry* pointer;
	};
}