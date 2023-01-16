#pragma once
#include <array>
#include <emmintrin.h>
#include "../../D3D/enums.h"

namespace Tracing
{
	using DirectCompute::eDataType;

	// File header of the trace file
	struct sFileHeader
	{
		static constexpr uint32_t correctMagic = 0xE6B4A12Du;	// random.org

		uint32_t magic;
		uint8_t formatVersion;
		uint8_t zzPadding;
		uint16_t cbItem;
		uint32_t countItems;
		uint32_t zzPadding2;
		uint64_t bytesPayload;
		uint32_t countStrings, bytesStrings;
	};
	// Payload data starts immediately after the header, bytesPayload bytes in total.
	// Then `bytesStrings` with string names, first countStrings * 4 of them are offsets, then ( bytesStrings - countStrings * 4 ) bytes with the string data.
	// The strings in the file are null-terminated.
	// Immediately after the strings, the next `cbItem` * `countItems` bytes are actual items (tensors and vectors) saved in the trace.
	// The format is weird because optimized for streaming.
	// These traces can grow large, we can’t afford memory keeping the payload data in memory.
	// Metadata is tiny compared to payload, we accumulate that in memory, and write to the end of the file when closed.

	enum struct eItemType : uint8_t
	{
		Buffer = 1,
		Tensor = 2,
	};

	struct sTraceItem
	{
		uint64_t payloadOffset;
		uint64_t payloadSize;
		std::array<uint32_t, 4> size;
		std::array<uint32_t, 4> stride;
		std::array<uint32_t, 4> formatArgs;
		eItemType itemType;
		eDataType dataType;
		uint8_t countFormatArgs = 0;
		uint8_t zzPadding = 0;
		uint32_t stringIndex;

		uint64_t buffer( uint64_t off, size_t length, eDataType type );

		uint64_t tensor( uint64_t off, __m128i ne, __m128i nb, eDataType type );
	};
}