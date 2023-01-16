#include "stdafx.h"
#include "logger.h"
#include "miscUtils.h"

namespace
{
	using namespace Whisper;

	// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
	// Lowest is red, middle is yellow, highest is green.
	static const std::array<const char*, 10> k_colors =
	{
		"\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
		"\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
	};

	static int colorIndex( const sToken& tok )
	{
		const float p = tok.probability;
		const float p3 = p * p * p;
		int col = (int)( p3 * float( k_colors.size() ) );
		col = std::max( 0, std::min( (int)k_colors.size() - 1, col ) );
		return col;
	}
}

void printTimeStamp( CStringA& rdi, Whisper::sTimeSpan ts )
{
	sTimeSpanFields fields = ts;
	uint32_t msec = fields.ticks / 10'000;
	uint32_t hr = fields.days * 24 + fields.hours;
	uint32_t min = fields.minutes;
	uint32_t sec = fields.seconds;
	rdi.AppendFormat( "%02d:%02d:%02d.%03d", hr, min, sec, msec );
}

HRESULT logNewSegments( const iTranscribeResult* results, size_t newSegments, bool printSpecial )
{
	sTranscribeLength length;
	CHECK( results->getSize( length ) );

	const size_t len = length.countSegments;
	size_t i = len - newSegments;

	const sSegment* const segments = results->getSegments();
	const sToken* const tokens = results->getTokens();

	CStringA str;
	for( ; i < len; i++ )
	{
		const sSegment& seg = segments[ i ];
		str = "[";
		printTimeStamp( str, seg.time.begin );
		str += " --> ";
		printTimeStamp( str, seg.time.end );
		str += "]  ";

		for( uint32_t j = 0; j < seg.countTokens; j++ )
		{
			const sToken& tok = tokens[ seg.firstToken + j ];
			if( !printSpecial && ( tok.flags & eTokenFlags::Special ) )
				continue;
			str += k_colors[ colorIndex( tok ) ];
			str += tok.text;
			str += "\033[0m";
		}
		logInfo( u8"%s", cstr( str ) );
	}

	return S_OK;
}