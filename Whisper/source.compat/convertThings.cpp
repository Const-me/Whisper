#include "stdafx.h"
#if BUILD_BOTH_VERSIONS
#include "../API/iContext.cl.h"
#include "convertThings.h"
using namespace Whisper;

sFullParams makeNewParams( const whisper_full_params& wfp )
{
	assert( nullptr == wfp.encoder_begin_callback );
	assert( nullptr == wfp.new_segment_callback );

	sFullParams res;
	memset( &res, 0, sizeof( res ) );

	res.strategy = (eSamplingStrategy)wfp.strategy;
	res.cpuThreads = wfp.n_threads;
	res.n_max_text_ctx = wfp.n_max_text_ctx;
	res.offset_ms = wfp.offset_ms;
	res.duration_ms = wfp.duration_ms;

	// flags
	uint32_t flags = 0;
	if( wfp.translate ) flags |= (uint32_t)eFullParamsFlags::Translate;
	if( wfp.no_context ) flags |= (uint32_t)eFullParamsFlags::NoContext;
	if( wfp.single_segment ) flags |= (uint32_t)eFullParamsFlags::SingleSegment;
	if( wfp.print_special ) flags |= (uint32_t)eFullParamsFlags::PrintSpecial;
	if( wfp.print_progress ) flags |= (uint32_t)eFullParamsFlags::PrintProgress;
	if( wfp.print_realtime ) flags |= (uint32_t)eFullParamsFlags::PrintRealtime;
	if( wfp.print_timestamps ) flags |= (uint32_t)eFullParamsFlags::PrintTimestamps;
	if( wfp.token_timestamps ) flags |= (uint32_t)eFullParamsFlags::TokenTimestamps;
	if( wfp.speed_up ) flags |= (uint32_t)eFullParamsFlags::SpeedupAudio;
	res.flags = (eFullParamsFlags)flags;

	res.language = findLanguageKeyA( wfp.language );
	res.thold_pt = wfp.thold_pt;
	res.thold_ptsum = wfp.thold_ptsum;
	res.max_len = wfp.max_len;
	res.greedy.n_past = wfp.greedy.n_past;
	res.beam_search.n_past = wfp.beam_search.n_past;
	res.beam_search.beam_width = wfp.beam_search.beam_width;
	res.beam_search.n_best = wfp.beam_search.n_best;
	res.audio_ctx = wfp.audio_ctx;
	res.prompt_tokens = wfp.prompt_tokens;
	res.prompt_n_tokens = wfp.prompt_n_tokens;

	return res;
}

namespace
{
	class NewParamsTemp
	{
		char language[ 5 ];
		iContext* newContext;
		pfnNewSegment newSegment;
		pfnEncoderBegin encoderBegin;

		static bool encBegin( struct whisper_context* ctx, void* user_data );
		static void newSeg( struct whisper_context* ctx, int n_new, void* user_data );

	public:

		void initialize( whisper_full_params& res, const Whisper::sFullParams& rsi, Whisper::iContext* context )
		{
			*(uint32_t*)( &language[ 0 ] ) = rsi.language;
			language[ 4 ] = '\0';
			res.language = language;

			newContext = context;

			if( nullptr != rsi.encoder_begin_callback )
			{
				encoderBegin = rsi.encoder_begin_callback;
				res.encoder_begin_callback = &encBegin;
				res.encoder_begin_callback_user_data = rsi.encoder_begin_callback_user_data;
			}
			else
			{
				encoderBegin = nullptr;
				res.encoder_begin_callback = nullptr;
				res.encoder_begin_callback_user_data = nullptr;
			}

			if( nullptr != rsi.new_segment_callback )
			{
				newSegment = rsi.new_segment_callback;
				res.new_segment_callback = &newSeg;
				res.new_segment_callback_user_data = rsi.new_segment_callback_user_data;
			}
			else
			{
				newSegment = nullptr;
				res.new_segment_callback = nullptr;
				res.new_segment_callback_user_data = nullptr;
			}
		}
	};

	static thread_local NewParamsTemp npTemp;

	bool NewParamsTemp::encBegin( struct whisper_context* ctx, void* user_data )
	{
		const NewParamsTemp& tmp = npTemp;
		HRESULT hr = tmp.encoderBegin( tmp.newContext, user_data );
		if( SUCCEEDED( hr ) )
			return S_OK == hr;
		throw hr;
	}

	void NewParamsTemp::newSeg( struct whisper_context* ctx, int n_new, void* user_data )
	{
		assert( n_new >= 0 );
		const NewParamsTemp& tmp = npTemp;
		HRESULT hr = tmp.newSegment( tmp.newContext, (uint32_t)n_new, user_data );
		if( SUCCEEDED( hr ) )
			return;
		throw hr;
	}
}

whisper_full_params makeOldParams( const Whisper::sFullParams& rsi, Whisper::iContext* context )
{
	whisper_full_params res;
	memset( &res, 0, sizeof( res ) );

	res.strategy = (whisper_sampling_strategy)rsi.strategy;
	res.n_threads = rsi.cpuThreads;
	res.n_max_text_ctx = rsi.n_max_text_ctx;
	res.offset_ms = rsi.offset_ms;
	res.duration_ms = rsi.duration_ms;

	// flags
	const uint32_t flags = (uint32_t)rsi.flags;
	auto hasFlag = [ = ]( eFullParamsFlags bit ) { return 0 != ( flags & (uint32_t)bit ); };

	res.translate = hasFlag( eFullParamsFlags::Translate );
	res.no_context = hasFlag( eFullParamsFlags::NoContext );
	res.single_segment = hasFlag( eFullParamsFlags::SingleSegment );
	res.print_special = hasFlag( eFullParamsFlags::PrintSpecial );
	res.print_progress = hasFlag( eFullParamsFlags::PrintProgress );
	res.print_realtime = hasFlag( eFullParamsFlags::PrintRealtime );
	res.print_timestamps = hasFlag( eFullParamsFlags::PrintTimestamps );
	res.token_timestamps = hasFlag( eFullParamsFlags::TokenTimestamps );
	res.speed_up = hasFlag( eFullParamsFlags::SpeedupAudio );

	res.thold_pt = rsi.thold_pt;
	res.thold_ptsum = rsi.thold_ptsum;
	res.max_len = rsi.max_len;
	res.greedy.n_past = rsi.greedy.n_past;
	res.beam_search.n_past = rsi.beam_search.n_past;
	res.beam_search.beam_width = rsi.beam_search.beam_width;
	res.beam_search.n_best = rsi.beam_search.n_best;
	res.audio_ctx = rsi.audio_ctx;
	res.prompt_tokens = rsi.prompt_tokens;
	res.prompt_n_tokens = rsi.prompt_n_tokens;

	NewParamsTemp& tmp = npTemp;
	tmp.initialize( res, rsi, context );
	return res;
}

#include "../Whisper/TranscribeResult.h"
#include <mfapi.h>

namespace
{
	inline sTimeSpan time( int64_t wt )
	{
		int64_t ticks = MFllMulDiv( wt, 10'000'000, 100, 0 );
		return sTimeSpan{ (uint64_t)ticks };
	}

	void makeNewResults( whisper_context* ctx, Whisper::eResultFlags flags, TranscribeResult& res )
	{
		const bool makeTokens = 0 != ( flags & eResultFlags::Tokens );
		res.segments.clear();
		res.tokens.clear();

		const int countSegments = whisper_full_n_segments( ctx );
		res.segments.resize( countSegments );
		const int tokenEot = whisper_token_eot( ctx );
		for( int i = 0; i < countSegments; i++ )
		{
			sSegment& seg = res.segments[ i ];
			seg.text = whisper_full_get_segment_text( ctx, i );
			seg.time.begin = time( whisper_full_get_segment_t0( ctx, i ) );
			seg.time.end = time( whisper_full_get_segment_t1( ctx, i ) );

			seg.firstToken = (uint32_t)res.tokens.size();
			seg.countTokens = 0;
			if( !makeTokens )
				continue;

			const int countTokens = whisper_full_n_tokens( ctx, i );
			seg.countTokens = countTokens;
			res.tokens.resize( res.tokens.size() + countTokens );
			for( int t = 0; t < countTokens; t++ )
			{
				sToken& tok = res.tokens[ seg.firstToken + t ];
				tok.text = whisper_full_get_token_text( ctx, i, t );

				const whisper_token_data src = whisper_full_get_token_data( ctx, i, t );
				tok.time.begin = time( src.t0 );
				tok.time.end = time( src.t1 );
				tok.probability = src.p;
				tok.probabilityTimestamp = src.pt;
				tok.ptsum = src.ptsum;
				tok.vlen = src.vlen;
				tok.id = src.id;
				uint32_t flags = 0;
				if( src.id >= tokenEot )
					flags |= eTokenFlags::Special;
				tok.flags = (eTokenFlags)flags;
			}
		}
	}
}

HRESULT makeNewResults( whisper_context* ctx, Whisper::eResultFlags flags, Whisper::iTranscribeResult** pp )
{
	static TranscribeResultStatic trs;
	if( flags & eResultFlags::NewObject )
	{
		return E_NOTIMPL;
	}
	else
	{
		makeNewResults( ctx, flags, trs );
		*pp = &trs;
		( *pp )->AddRef();
		return S_OK;
	}
}
#endif