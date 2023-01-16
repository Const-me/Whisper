#pragma once
#include <stdint.h>
#include <array>
#ifndef GUID_DEFINED
#define GUID_DEFINED
#endif

struct GUID
{
	uint32_t Data1;
	uint16_t Data2;
	uint16_t Data3;
	std::array<uint8_t, 8> Data4;

	constexpr inline bool operator==( const GUID& that ) const
	{
		return Data1 == that.Data1 && Data2 == that.Data2 && Data3 == that.Data3 && Data4 == that.Data4;
	}
};

using REFIID = const GUID&;