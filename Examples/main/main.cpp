#include "params.h"
#include "../../Whisper/API/iContext.cl.h"
#include "../../Whisper/API/iMediaFoundation.cl.h"
#include "../../ComLightLib/comLightClient.h"
#include "miscUtils.h"
#include <array>
#include <atomic>
#include "textWriter.h"
using namespace Whisper;

#define STREAM_AUDIO 1

static HRESULT loadWhisperModel( const wchar_t* path, const std::wstring& gpu, iModel** pp )
{
	using namespace Whisper;
	sModelSetup setup;
	setup.impl = eModelImplementation::GPU;
	if( !gpu.empty() )
		setup.adapter = gpu.c_str();
	return Whisper::loadModel( path, setup, nullptr, pp );
}

namespace
{
	// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
	// Lowest is red, middle is yellow, highest is green.
	static const std::array<const char*, 10> k_colors =
	{
		"\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
		"\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
	};

	std::string to_timestamp( sTimeSpan ts, bool comma = false )
	{
		sTimeSpanFields fields = ts;
		uint32_t msec = fields.ticks / 10'000;
		uint32_t hr = fields.days * 24 + fields.hours;
		uint32_t min = fields.minutes;
		uint32_t sec = fields.seconds;

		char buf[ 32 ];
		snprintf( buf, sizeof( buf ), "%02d:%02d:%02d%s%03d", hr, min, sec, comma ? "," : ".", msec );
		return std::string( buf );
	}

	static int colorIndex( const sToken& tok )
	{
		const float p = tok.probability;
		const float p3 = p * p * p;
		int col = (int)( p3 * float( k_colors.size() ) );
		col = std::max( 0, std::min( (int)k_colors.size() - 1, col ) );
		return col;
	}

	HRESULT __cdecl newSegmentCallback( iContext* context, uint32_t n_new, void* user_data ) noexcept
	{
		ComLight::CComPtr<iTranscribeResult> results;
		CHECK( context->getResults( eResultFlags::Timestamps | eResultFlags::Tokens, &results ) );

		sTranscribeLength length;
		CHECK( results->getSize( length ) );

		const whisper_params& params = *( (const whisper_params*)user_data );

		// print the last n_new segments
		const uint32_t s0 = length.countSegments - n_new;
		if( s0 == 0 )
			printf( "\n" );

		const sSegment* const segments = results->getSegments();
		const sToken* const tokens = results->getTokens();

		for( uint32_t i = s0; i < length.countSegments; i++ )
		{
			const sSegment& seg = segments[ i ];

			if( params.no_timestamps )
			{
				if( params.print_colors )
				{
					for( uint32_t j = 0; j < seg.countTokens; j++ )
					{
						const sToken& tok = tokens[ seg.firstToken + j ];
						if( !params.print_special && ( tok.flags & eTokenFlags::Special ) )
							continue;
						wprintf( L"%S%s%S", k_colors[ colorIndex( tok ) ], utf16( tok.text ).c_str(), "\033[0m" );
					}
				}
				else
					wprintf( L"%s", utf16( seg.text ).c_str() );
				fflush( stdout );
				continue;
			}

			std::string speaker = "";

			if( params.diarize )
			{
				eSpeakerChannel channel;
				HRESULT hr = context->detectSpeaker( seg.time, channel );
				if( SUCCEEDED( hr ) && channel != eSpeakerChannel::NoStereoData )
				{
					using namespace std::string_literals;
					switch( channel )
					{
					case eSpeakerChannel::Unsure:
						speaker = "(speaker ?)"s;
						break;
					case eSpeakerChannel::Left:
						speaker = "(speaker 0)"s;
						break;
					case eSpeakerChannel::Right:
						speaker = "(speaker 1)";
						break;
					}
				}
			}

			if( params.print_colors )
			{
				printf( "[%s --> %s] %s ",
					to_timestamp( seg.time.begin ).c_str(),
					to_timestamp( seg.time.end ).c_str(),
					speaker.c_str() );

				for( uint32_t j = 0; j < seg.countTokens; j++ )
				{
					const sToken& tok = tokens[ seg.firstToken + j ];
					if( !params.print_special && ( tok.flags & eTokenFlags::Special ) )
						continue;
					wprintf( L"%S%s%S", k_colors[ colorIndex( tok ) ], utf16( tok.text ).c_str(), "\033[0m" );
				}
				printf( "\n" );
			}
			else
				wprintf( L"[%S --> %S]  %S%s\n", to_timestamp( seg.time.begin ).c_str(), to_timestamp( seg.time.end ).c_str(), speaker.c_str(), utf16( seg.text ).c_str() );
		}
		return S_OK;
	}

	HRESULT __cdecl beginSegmentCallback( iContext* context, void* user_data ) noexcept
	{
		std::atomic_bool* flag = (std::atomic_bool*)user_data;
		bool aborted = flag->load();
		return aborted ? S_FALSE : S_OK;
	}

	HRESULT setupConsoleColors()
	{
		HANDLE h = GetStdHandle( STD_OUTPUT_HANDLE );
		if( h == INVALID_HANDLE_VALUE )
			return HRESULT_FROM_WIN32( GetLastError() );

		DWORD mode = 0;
		if( !GetConsoleMode( h, &mode ) )
			return HRESULT_FROM_WIN32( GetLastError() );
		if( 0 != ( mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING ) )
			return S_FALSE;

		mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
		if( !SetConsoleMode( h, mode ) )
			return HRESULT_FROM_WIN32( GetLastError() );
		return S_OK;
	}
}

static void __stdcall setPrompt( const int* ptr, int length, void* pv )
{
	std::vector<int>& vec = *( std::vector<int> * )( pv );
	if( length > 0 )
		vec.assign( ptr, ptr + length );
}

int wmain( int argc, wchar_t* argv[] )
{
	// Whisper::dbgCompareTraces( LR"(C:\Temp\2remove\Whisper\ref.bin)", LR"(C:\Temp\2remove\Whisper\gpu.bin )" ); return 0;

	// Tell logger to use the standard output stream for the messages
	{
		Whisper::sLoggerSetup logSetup;
		logSetup.flags = eLoggerFlags::UseStandardError;
		logSetup.level = eLogLevel::Debug;
		Whisper::setupLogger( logSetup );
	}

	whisper_params params;
	if( !params.parse( argc, argv ) )
		return 1;

	if( params.print_colors )
	{
		if( FAILED( setupConsoleColors() ) )
			params.print_colors = false;
	}

	if( params.fname_inp.empty() )
	{
		fprintf( stderr, "error: no input files specified\n" );
		whisper_print_usage( argc, argv, params );
		return 2;
	}

	if( Whisper::findLanguageKeyA( params.language.c_str() ) == UINT_MAX )
	{
		fprintf( stderr, "error: unknown language '%s'\n", params.language.c_str() );
		whisper_print_usage( argc, argv, params );
		return 3;
	}

	ComLight::CComPtr<iModel> model;
	HRESULT hr = loadWhisperModel( params.model.c_str(), params.gpu, &model );
	if( FAILED( hr ) )
	{
		printError( "failed to load the model", hr );
		return 4;
	}

	std::vector<int> prompt;
	if( !params.prompt.empty() )
	{
		hr = model->tokenize( params.prompt.c_str(), &setPrompt, &prompt );
		if( FAILED( hr ) )
		{
			printError( "failed to tokenize the initial prompt", hr );
			return 5;
		}
	}

	ComLight::CComPtr<iContext> context;
	hr = model->createContext( &context );
	if( FAILED( hr ) )
	{
		printError( "failed to initialize whisper context", hr );
		return 6;
	}

	ComLight::CComPtr<iMediaFoundation> mf;
	hr = initMediaFoundation( &mf );
	if( FAILED( hr ) )
	{
		printError( "failed to initialize Media Foundation runtime", hr );
		return 7;
	}

	for( const std::wstring& fname : params.fname_inp )
	{
		// print some info about the processing
		{
			if( model->isMultilingual() == S_FALSE )
			{
				if( params.language != "en" || params.translate )
				{
					params.language = "en";
					params.translate = false;
					fprintf( stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__ );
				}
			}
		}

		// run the inference
		Whisper::sFullParams wparams;
		context->fullDefaultParams( eSamplingStrategy::Greedy, &wparams );

		wparams.resetFlag( eFullParamsFlags::PrintRealtime | eFullParamsFlags::PrintProgress );
		wparams.setFlag( eFullParamsFlags::PrintTimestamps, !params.no_timestamps );
		wparams.setFlag( eFullParamsFlags::PrintSpecial, params.print_special );
		wparams.setFlag( eFullParamsFlags::Translate, params.translate );
		// When there're multiple input files, assuming they're independent clips
		wparams.setFlag( eFullParamsFlags::NoContext );
		wparams.language = Whisper::makeLanguageKey( params.language.c_str() );
		wparams.cpuThreads = params.n_threads;
		if( params.max_context != UINT_MAX )
			wparams.n_max_text_ctx = params.max_context;
		wparams.offset_ms = params.offset_t_ms;
		wparams.duration_ms = params.duration_ms;

		wparams.setFlag( eFullParamsFlags::TokenTimestamps, params.output_wts || params.max_len > 0 );
		wparams.thold_pt = params.word_thold;
		wparams.max_len = params.output_wts && params.max_len == 0 ? 60 : params.max_len;

		wparams.setFlag( eFullParamsFlags::SpeedupAudio, params.speed_up );

		if( !prompt.empty() )
		{
			wparams.prompt_tokens = prompt.data();
			wparams.prompt_n_tokens = (int)prompt.size();
		}

		// This callback is called on each new segment
		if( !wparams.flag( eFullParamsFlags::PrintRealtime ) )
		{
			wparams.new_segment_callback = &newSegmentCallback;
			wparams.new_segment_callback_user_data = &params;
		}

		// example for abort mechanism
		// in this example, we do not abort the processing, but we could if the flag is set to true
		// the callback is called before every encoder run - if it returns false, the processing is aborted
		std::atomic_bool is_aborted = false;
		{
			wparams.encoder_begin_callback = &beginSegmentCallback;
			wparams.encoder_begin_callback_user_data = &is_aborted;
		}

		if( STREAM_AUDIO && !wparams.flag( eFullParamsFlags::TokenTimestamps ) )
		{
			ComLight::CComPtr<iAudioReader> reader;
			CHECK( mf->openAudioFile( fname.c_str(), params.diarize, &reader ) );
			sProgressSink progressSink{ nullptr, nullptr };
			hr = context->runStreamed( wparams, progressSink, reader );
		}
		else
		{
			// Token-level timestamps feature is not currently implemented when streaming the audio
			// When these timestamps are requested, fall back to buffered mode.
			ComLight::CComPtr<iAudioBuffer> buffer;
			CHECK( mf->loadAudioFile( fname.c_str(), params.diarize, &buffer ) );
			hr = context->runFull( wparams, buffer );
		}

		if( FAILED( hr ) )
		{
			printError( "Unable to process audio", hr );
			return 10;
		}

		if( params.output_txt )
		{
			bool timestamps = !params.no_timestamps;
			hr = writeText( context, fname.c_str(), timestamps );
			if( FAILED( hr ) )
				printError( "Unable to produce the text file", hr );
		}

		if( params.output_srt )
		{
			hr = writeSubRip( context, fname.c_str() );
			if( FAILED( hr ) )
				printError( "Unable to produce the text file", hr );
		}

		if( params.output_vtt )
		{
			hr = writeWebVTT( context, fname.c_str() );
			if( FAILED( hr ) )
				printError( "Unable to produce the text file", hr );
		}
	}

	context->timingsPrint();
	context = nullptr;
	return 0;
}