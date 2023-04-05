#include "params.h"
#include <algorithm>
#include <thread>
#include "miscUtils.h"
#include "../../Whisper/API/iContext.cl.h"

whisper_params::whisper_params()
{
#ifdef _DEBUG
	n_threads = 2;
#else
	n_threads = std::min( 4u, std::thread::hardware_concurrency() );
#endif	
}

namespace
{
	const char* cstr( bool b )
	{
		return b ? "true" : "false";
	}
}

void whisper_print_usage( int argc, wchar_t** argv, const whisper_params& params )
{
	fprintf( stderr, "\n" );
	fprintf( stderr, "usage: %S [options] file0.wav file1.wav ...\n", argv[ 0 ] );
	fprintf( stderr, "\n" );
	fprintf( stderr, "options:\n" );
	fprintf( stderr, "  -h,       --help          [default] show this help message and exit\n" );
	fprintf( stderr, "  -la,      --list-adapters List graphic adapters and exit\n" );
	fprintf( stderr, "  -gpu,     --use-gpu       The graphic adapter to use for inference\n" );
	fprintf( stderr, "  -t N,     --threads N     [%-7d] number of threads to use during computation\n", params.n_threads );
	fprintf( stderr, "  -p N,     --processors N  [%-7d] number of processors to use during computation\n", params.n_processors );
	fprintf( stderr, "  -ot N,    --offset-t N    [%-7d] time offset in milliseconds\n", params.offset_t_ms );
	fprintf( stderr, "  -on N,    --offset-n N    [%-7d] segment index offset\n", params.offset_n );
	fprintf( stderr, "  -d  N,    --duration N    [%-7d] duration of audio to process in milliseconds\n", params.duration_ms );
	fprintf( stderr, "  -mc N,    --max-context N [%-7d] maximum number of text context tokens to store\n", params.max_context );
	fprintf( stderr, "  -ml N,    --max-len N     [%-7d] maximum segment length in characters\n", params.max_len );
	fprintf( stderr, "  -wt N,    --word-thold N  [%-7.2f] word timestamp probability threshold\n", params.word_thold );
	fprintf( stderr, "  -su,      --speed-up      [%-7s] speed up audio by x2 (reduced accuracy)\n", cstr( params.speed_up ) );
	fprintf( stderr, "  -tr,      --translate     [%-7s] translate from source language to english\n", cstr( params.translate ) );
	fprintf( stderr, "  -di,      --diarize       [%-7s] stereo audio diarization\n", cstr( params.diarize ) );
	fprintf( stderr, "  -otxt,    --output-txt    [%-7s] output result in a text file\n", cstr( params.output_txt ) );
	fprintf( stderr, "  -ovtt,    --output-vtt    [%-7s] output result in a vtt file\n", cstr( params.output_vtt ) );
	fprintf( stderr, "  -osrt,    --output-srt    [%-7s] output result in a srt file\n", cstr( params.output_srt ) );
	fprintf( stderr, "  -owts,    --output-words  [%-7s] output script for generating karaoke video\n", cstr( params.output_wts ) );
	fprintf( stderr, "  -ps,      --print-special [%-7s] print special tokens\n", cstr( params.print_special ) );
	fprintf( stderr, "  -nc,      --no-colors     [%-7s] do not print colors\n", cstr( !params.print_colors ) );
	fprintf( stderr, "  -nt,      --no-timestamps [%-7s] do not print timestamps\n", cstr( params.no_timestamps ) );
	fprintf( stderr, "  -l LANG,  --language LANG [%-7s] spoken language\n", params.language.c_str() );
	fprintf( stderr, "  -m FNAME, --model FNAME   [%-7S] model path\n", params.model.c_str() );
	fprintf( stderr, "  -f FNAME, --file FNAME    [%-7s] path of the input audio file\n", "" );
	fprintf( stderr, "  --prompt                            initial prompt for the model\n" );
	fprintf( stderr, "\n" );
}

static void __stdcall pfnListAdapter( const wchar_t* name, void* )
{
	wprintf( L"\"%s\"\n", name );
}

static void listGpus()
{
	printf( "    Available graphic adapters:\n" );
	HRESULT hr = Whisper::listGPUs( &pfnListAdapter, nullptr );
	if( SUCCEEDED( hr ) )
		return;
	printError( "Unable to enumerate GPUs", hr );
}

bool whisper_params::parse( int argc, wchar_t* argv[] )
{
	for( int i = 1; i < argc; i++ )
	{
		std::wstring arg = argv[ i ];

		if( arg[ 0 ] != '-' )
		{
			fname_inp.push_back( arg );
			continue;
		}

		if( arg == L"-h" || arg == L"--help" )
		{
			whisper_print_usage( argc, argv, *this );
			return false;
		}

		if( arg == L"-la" || arg == L"--list-adapters" )
		{
			listGpus();
			return false;
		}

		else if( arg == L"-t" || arg == L"--threads" ) { n_threads = std::stoul( argv[ ++i ] ); }
		else if( arg == L"-p" || arg == L"--processors" ) { n_processors = std::stoul( argv[ ++i ] ); }
		else if( arg == L"-ot" || arg == L"--offset-t" ) { offset_t_ms = std::stoul( argv[ ++i ] ); }
		else if( arg == L"-on" || arg == L"--offset-n" ) { offset_n = std::stoul( argv[ ++i ] ); }
		else if( arg == L"-d" || arg == L"--duration" ) { duration_ms = std::stoul( argv[ ++i ] ); }
		else if( arg == L"-mc" || arg == L"--max-context" ) { max_context = std::stoul( argv[ ++i ] ); }
		else if( arg == L"-ml" || arg == L"--max-len" ) { max_len = std::stoul( argv[ ++i ] ); }
		else if( arg == L"-wt" || arg == L"--word-thold" ) { word_thold = std::stof( argv[ ++i ] ); }
		else if( arg == L"-su" || arg == L"--speed-up" ) { speed_up = true; }
		else if( arg == L"-tr" || arg == L"--translate" ) { translate = true; }
		else if( arg == L"-di" || arg == L"--diarize" ) { diarize = true; }
		else if( arg == L"-otxt" || arg == L"--output-txt" ) { output_txt = true; }
		else if( arg == L"-ovtt" || arg == L"--output-vtt" ) { output_vtt = true; }
		else if( arg == L"-osrt" || arg == L"--output-srt" ) { output_srt = true; }
		else if( arg == L"-owts" || arg == L"--output-words" ) { output_wts = true; }
		else if( arg == L"-ps" || arg == L"--print-special" ) { print_special = true; }
		else if( arg == L"-nc" || arg == L"--no-colors" ) { print_colors = false; }
		else if( arg == L"-nt" || arg == L"--no-timestamps" ) { no_timestamps = true; }
		else if( arg == L"-l" || arg == L"--language" ) { language = utf8( argv[ ++i ] ); }
		else if( arg == L"-m" || arg == L"--model" ) { model = argv[ ++i ]; }
		else if( arg == L"-f" || arg == L"--file" ) { fname_inp.push_back( argv[ ++i ] ); }
		else if( arg == L"-gpu" || arg == L"--use-gpu" ) { gpu = argv[ ++i ]; }
		else if( arg == L"--prompt" ) { prompt = utf8( argv[ ++i ] ); }
		else
		{
			fprintf( stderr, "error: unknown argument: %S\n", arg.c_str() );
			whisper_print_usage( argc, argv, *this );
			return false;
		}
	}
	return true;
}