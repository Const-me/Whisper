#pragma once
#include <stdint.h>
#include <array>

namespace DirectCompute
{
	struct LookupTablesData
	{
		std::array<uint16_t, 0x10000> gelu;
		std::array<uint16_t, 0x10000> exponent;

		LookupTablesData();
	};
}