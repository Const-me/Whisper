#pragma once
#include <stdint.h>
#include <assert.h>

namespace Whisper
{
	enum struct eModelImplementation : uint32_t
	{
		GPU = 1,
		Hybrid = 2,
		Reference = 3,
	};

	struct sTimeSpanFields
	{
		uint32_t days;
		uint8_t hours, minutes, seconds;
		uint32_t ticks;

		sTimeSpanFields( uint64_t tt )
		{
			ticks = (uint32_t)( tt % 10'000'000 );
			tt /= 10'000'000;
			seconds = (uint8_t)( tt % 60 );
			tt /= 60;
			minutes = (uint8_t)( tt % 60 );
			tt /= 60;
			hours = (uint8_t)( tt % 24 );
			tt /= 24;
			days = (uint32_t)tt;
		}
	};

	struct sTimeSpan
	{
		uint64_t ticks;

		operator sTimeSpanFields() const
		{
			return sTimeSpanFields{ ticks };
		}
		void operator=( uint64_t tt )
		{
			ticks = tt;
		}
		void operator=( int64_t tt )
		{
			assert( tt >= 0 );
			ticks = (uint64_t)tt;
		}
	};

	// Start and end times of the segment or token, expressed in 100-nanosecond ticks
	struct sTimeInterval
	{
		sTimeSpan begin, end;
	};

	// Segment data
	struct sSegment
	{
		// Segment text, null-terminated, and probably UTF-8 encoded
		const char* text;
		// Start and end times of the segment
		sTimeInterval time;
		uint32_t firstToken, countTokens;
	};

	enum eTokenFlags : uint32_t
	{
		None = 0,
		Special = 1,
	};
	inline bool operator &( eTokenFlags a, eTokenFlags b )
	{
		return 0 != ( (uint32_t)a & (uint32_t)b );
	}

	// Token data
	struct sToken
	{
		// Token text, null-terminated, and probably UTF-8 encoded
		const char* text;
		// Start and end times of the token
		sTimeInterval time;
		// Probability of the token
		float probability;
		// Probability of the timestamp token
		float probabilityTimestamp;
		// Sum of probabilities of all timestamp tokens
		float ptsum;
		// Voice length of the token
		float vlen;
		// Token id
		int id;
		eTokenFlags flags;
	};

	struct sTranscribeLength
	{
		uint32_t countSegments, countTokens;
	};

	enum struct eResultFlags : uint32_t
	{
		None = 0,
		// Return individual tokens in addition to the segments
		Tokens = 1,
		// Return timestamps
		Timestamps = 2,

		// Create a new COM object for the results.
		// Without this flag, the context returns a pointer to the COM object stored in the context.
		// The content of that object is replaced every time you call iContext.getResults method
		NewObject = 0x100,
	};

	inline eResultFlags operator |( eResultFlags a, eResultFlags b )
	{
		return (eResultFlags)( (uint32_t)a | (uint32_t)b );
	}

	inline bool operator &( eResultFlags a, eResultFlags b )
	{
		return 0 != ( (uint32_t)a & (uint32_t)b );
	}
}