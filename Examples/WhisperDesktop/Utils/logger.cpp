#include "stdafx.h"
#include "logger.h"
#include "miscUtils.h"
#include <string>

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

void printTime( CStringA& rdi, Whisper::sTimeSpan time, bool comma )
{
	Whisper::sTimeSpanFields fields = time;
	const uint32_t hours = fields.days * 24 + fields.hours;
	const char separator = comma ? ',' : '.';
	rdi.AppendFormat( "%02d:%02d:%02d%c%03d",
		(int)hours,
		(int)fields.minutes,
		(int)fields.seconds,
		separator,
		fields.ticks / 10'000 );
}

bool utf8_check_is_valid(const char *str, int len) {
	// based on https://gist.github.com/ichramm/3ffeaf7ba4f24853e9ecaf176da84566
    int n;
    for (int i = 0; i < len; ++i) {
        unsigned char c = (unsigned char) str[i];
        //if (c==0x09 || c==0x0a || c==0x0d || (0x20 <= c && c <= 0x7e) ) n = 0; // is_printable_ascii
        if (0x00 <= c && c <= 0x7f) {
            n=0; // 0bbbbbbb
        } else if ((c & 0xE0) == 0xC0) {
            n=1; // 110bbbbb
        } else if ( c==0xed && i<(len-1) && ((unsigned char)str[i+1] & 0xa0)==0xa0) {
            return false; //U+d800 to U+dfff
        } else if ((c & 0xF0) == 0xE0) {
            n=2; // 1110bbbb
        } else if ((c & 0xF8) == 0xF0) {
            n=3; // 11110bbb
        //} else if (($c & 0xFC) == 0xF8) { n=4; // 111110bb //byte 5, unnecessary in 4 byte UTF-8
        //} else if (($c & 0xFE) == 0xFC) { n=5; // 1111110b //byte 6, unnecessary in 4 byte UTF-8
        } else {
            return false;
        }

        for (int j = 0; j < n && i < len; ++j) { // n bytes matching 10bbbbbb follow ?
            if ((++i == len) || (( (unsigned char)str[i] & 0xC0) != 0x80)) {
                return false;
            }
        }
    }
    return true;
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
	std::string buffer;
	for( ; i < len; i++ )
	{
		const sSegment& seg = segments[ i ];
		str = "[";
		printTime( str, seg.time.begin );
		str += " --> ";
		printTime( str, seg.time.end );
		str += "]  ";

		for( uint32_t j = 0; j < seg.countTokens; j++ )
		{
			const sToken& tok = tokens[ seg.firstToken + j ];
			if( !printSpecial && ( tok.flags & eTokenFlags::Special ) )
		        continue;
		    if (utf8_check_is_valid(tok.text, strlen(tok.text))) {
				str += k_colors[ colorIndex( tok ) ];
		        str += tok.text;
		        str += "\033[0m";
		    } else {
		        for (int k = 0; k < strlen(tok.text); k++) {
		            buffer.push_back(tok.text[k]);
		            if (utf8_check_is_valid(&buffer[0], buffer.size())) {
		                str += k_colors[ colorIndex( tok ) ];
		                str += &buffer[0];
		                str += "\033[0m";
		                buffer.clear();
		            }
		        }
		    }
		}
		logInfo( u8"%s", cstr( str ) );
	}

	return S_OK;
}