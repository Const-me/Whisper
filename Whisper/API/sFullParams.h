#pragma once
#include <stdint.h>
#include <assert.h>

namespace Whisper
{
	// Available sampling strategies
	enum struct eSamplingStrategy : int
	{
		// Always select the most probable token
		Greedy,
		// TODO: not implemented yet!
		BeamSearch,
	};

	using pfnNewSegment = HRESULT( __cdecl* )( iContext* ctx, uint32_t n_new, void* user_data ) noexcept;

	// Return S_OK to proceed, or S_FALSE to stop the process and return S_OK from runFull / runStreamed method
	using pfnEncoderBegin = HRESULT( __cdecl* )( iContext* ctx, void* user_data ) noexcept;

	enum struct eFullParamsFlags : uint32_t
	{
		Translate = 1,
		NoContext = 2,
		SingleSegment = 4,
		PrintSpecial = 8,
		PrintProgress = 0x10,
		PrintRealtime = 0x20,
		PrintTimestamps = 0x40,

		// Experimental
		TokenTimestamps = 0x100,
		SpeedupAudio = 0x200,
	};

	inline eFullParamsFlags operator | ( eFullParamsFlags a, eFullParamsFlags b )
	{
		return (eFullParamsFlags)( (uint32_t)a | (uint32_t)b );
	}
	inline void operator |= ( eFullParamsFlags& a, eFullParamsFlags b )
	{
		a = a | b;
	}

	struct sFullParams
	{
		eSamplingStrategy strategy;
		// Count of CPU threads
		int cpuThreads;
		int n_max_text_ctx;
		int offset_ms;          // start offset in ms
		int duration_ms;        // audio duration to process in ms
		eFullParamsFlags flags;
		uint32_t language;

		// [EXPERIMENTAL] token-level timestamps
		float thold_pt;         // timestamp token probability threshold (~0.01)
		float thold_ptsum;      // timestamp token sum probability threshold (~0.01)
		int   max_len;          // max segment length in characters
		int   max_tokens;       // max tokens per segment (0 = no limit)

		struct
		{
			int n_past;
		} greedy;

		struct
		{
			int n_past;
			int beam_width;
			int n_best;
		} beam_search;

		// [EXPERIMENTAL] speed-up techniques
		int  audio_ctx;         // overwrite the audio context size (0 = use default)

		// tokens to provide the whisper model as initial prompt
		// these are prepended to any existing text context from a previous call
		const whisper_token* prompt_tokens;
		int prompt_n_tokens;

		pfnNewSegment new_segment_callback;
		void* new_segment_callback_user_data;

		pfnEncoderBegin encoder_begin_callback;
		void* encoder_begin_callback_user_data;

		// Couple utility methods, they workaround the lack of bit fields in C++
		inline bool flag( eFullParamsFlags f ) const
		{
			return 0 != ( (uint32_t)flags & (uint32_t)f );
		}
		inline void resetFlag( eFullParamsFlags bit )
		{
			uint32_t f = (uint32_t)flags;
			f &= ~(uint32_t)bit;
			flags = (eFullParamsFlags)f;
		}
		inline void setFlag( eFullParamsFlags bit, bool set = true )
		{
			uint32_t f = (uint32_t)flags;
			if( set )
				f |= (uint32_t)bit;
			else
				f &= ~(uint32_t)bit;
			flags = (eFullParamsFlags)f;
		}
	};

	struct sSegmentTime
	{
		int64_t begin, end;
	};

	inline uint32_t makeLanguageKey( const char* code )
	{
		assert( strlen( code ) <= 4 );
		uint32_t res = 0;
		uint32_t shift = 0;
		for( size_t i = 0; i < 4; i++, code++, shift += 8 )
		{
			const char c = *code;
			if( c == '\0' )
				return res;
			uint32_t u32 = (uint8_t)c;
			u32 = u32 << shift;
			res |= u32;
		}
		return res;
	}

	using pfnReportProgress = HRESULT( __stdcall* )( double val, iContext* ctx, void* pv ) noexcept;
	struct sProgressSink
	{
		pfnReportProgress pfn;
		void* pv;
	};
}