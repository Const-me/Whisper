#pragma once
#include "../../Whisper/Utils/Trace/TraceStructures.h"
#include <atlstr.h>
#include <atlfile.h>

namespace Tracing
{
	class TraceReader
	{
		const uint8_t* payloadPointer = nullptr;
		const sTraceItem* items = nullptr;
		size_t countItems = 0;
		size_t countStrings = 0;
		const uint32_t* stringIndex = nullptr;
		const char* stringData = nullptr;

		CAtlFile file;
		CAtlFileMapping<uint8_t> mapping;

	public:

		TraceReader() = default;
		~TraceReader() = default;

		HRESULT open( LPCTSTR path );
		size_t size() const { return countItems; }
		const sTraceItem& operator[]( size_t idx ) const;
		CStringA getName( const sTraceItem& item ) const;

		const void* payload( const sTraceItem& item ) const
		{
			return payloadPointer + item.payloadOffset;
		}
	};
}