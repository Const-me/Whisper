#include "stdafx.h"
#include "ContextImpl.h"
#include <mfapi.h>
#include "MelStreamer.h"
#include "../API/iMediaFoundation.cl.h"
#include "../Utils/Trace/tracing.h"
using namespace Whisper;

static int getCpuCoresCount()
{
	DWORD bufferSize = 0;
	GetLogicalProcessorInformation( NULL, &bufferSize );

	// The SYSTEM_LOGICAL_PROCESSOR_INFORMATION structure has a uint64_t field
	// Ideally need to align by 8 bytes, and that's why uint64_t type for the storage
	std::unique_ptr<uint64_t[]> buffer = std::make_unique<uint64_t[]>( ( bufferSize + 7 ) / 8 );

	SYSTEM_LOGICAL_PROCESSOR_INFORMATION* ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)buffer.get();
	if( !GetLogicalProcessorInformation( ptr, &bufferSize ) )
	{
		HRESULT hr = getLastHr();
		logWarningHr( hr, u8"GetLogicalProcessorInformation" );
		return 0;
	}

	DWORD byteOffset = 0;
	int physicalCores = 0;
	while( byteOffset < bufferSize )
	{
		if( ptr->Relationship == RelationProcessorCore )
			physicalCores++;
		byteOffset += sizeof( SYSTEM_LOGICAL_PROCESSOR_INFORMATION );
		ptr++;
	}
	return physicalCores;
}

int ContextImpl::defaultThreadsCount() const
{
#if BUILD_HYBRID_VERSION
	const bool isHybrid = !model.shared->hybridTensors.layers.empty();
#else
	constexpr bool isHybrid = false;
#endif

	SYSTEM_INFO si;
	GetSystemInfo( &si );
	const int hardwareThreads = (int)si.dwNumberOfProcessors;

	if( !isHybrid )
		return std::min( hardwareThreads, 4 );

	// It seems the CPU decoder in the hybrid context doesn’t scale well with count of hardware threads, but it does scale with count of physical cores.
	int cores = getCpuCoresCount();
	if( cores > 1 )
		return cores;

	return hardwareThreads;
}

HRESULT COMLIGHTCALL ContextImpl::fullDefaultParams( eSamplingStrategy strategy, sFullParams* rdi )
{
	// whisper_full_default_params
	if( nullptr == rdi )
		return E_POINTER;
	memset( rdi, 0, sizeof( sFullParams ) );

	rdi->strategy = strategy;
	rdi->cpuThreads = defaultThreadsCount();
	rdi->n_max_text_ctx = 16384;
	rdi->flags = eFullParamsFlags::PrintProgress | eFullParamsFlags::PrintTimestamps;
	rdi->thold_pt = 0.01f;
	rdi->thold_ptsum = 0.01f;
	rdi->language = makeLanguageKey( "en" );

	switch( strategy )
	{
	case eSamplingStrategy::Greedy:
		rdi->beam_search.n_past = -1;
		rdi->beam_search.beam_width = -1;
		rdi->beam_search.n_best = -1;
		break;
	case eSamplingStrategy::BeamSearch:
		rdi->greedy.n_past = -1;
		rdi->beam_search.beam_width = 10;
		rdi->beam_search.n_best = 5;
		break;
	default:
		logError( u8"Unknown sampling strategy %i", (int)strategy );
		return E_INVALIDARG;
	}
	return S_OK;
}

HRESULT COMLIGHTCALL ContextImpl::getModel( iModel** pp )
{
	if( nullptr == pp )
		return E_POINTER;
	if( !modelPtr )
		return OLE_E_BLANK;
	*pp = modelPtr;
	modelPtr->AddRef();
	return S_OK;
}

size_t ContextImpl::Segment::memoryUsage() const
{
	return text.capacity() + vectorMemoryUse( tokens );
}

__m128i ContextImpl::getMemoryUse() const
{
	// Misc. system RAM
	size_t cb = vectorMemoryUse( result_all );
	for( const auto& r : result_all )
		cb += r.memoryUsage();
	cb += vectorMemoryUse( prompt_past );
	cb += vectorMemoryUse( energy );
	cb += vectorMemoryUse( probs );
	cb += vectorMemoryUse( probs_id );
	cb += vectorMemoryUse( results.segments );
	cb += vectorMemoryUse( results.tokens );
	cb += spectrogram.memoryUsage();

	__m128i res = setLow_size( cb );
	// Add all the VRAM in the temporary buffers
	res = _mm_add_epi64( res, context.getMemoryUse() );
	return res;
}

namespace
{
	struct PrintedSize
	{
		double val;
		const char* unit;
		PrintedSize( int64_t cb )
		{
			if( cb < ( 1 << 10 ) )
			{
				val = (double)cb;
				unit = "bytes";
			}
			else if( cb < ( 1 << 20 ) )
			{
				val = (double)cb * ( 1.0 / ( 1 << 10 ) );
				unit = "KB";
			}
			else if( cb < ( 1 << 30 ) )
			{
				val = (double)cb * ( 1.0 / ( 1 << 20 ) );
				unit = "MB";
			}
			else
			{
				val = (double)cb * ( 1.0 / ( 1 << 30 ) );
				unit = "GB";
			}
		}
	};

	static void __declspec( noinline ) logMemoryUse( const char* what, __m128i cb )
	{
		PrintedSize sys{ _mm_cvtsi128_si64( cb ) };
		PrintedSize vram{ _mm_extract_epi64( cb, 1 ) };
		logInfo( u8"%s\t%g %s RAM, %g %s VRAM", what, sys.val, sys.unit, vram.val, vram.unit );
	}
}

HRESULT COMLIGHTCALL ContextImpl::timingsPrint()
{
	profiler.print();

	auto ts = device.setForCurrentThread();
	const __m128i memModel = model.getMemoryUse();
	const __m128i memContext = getMemoryUse();
	logInfo( u8"    Memory Usage" );
	logMemoryUse( "Model", memModel );
	logMemoryUse( "Context", memContext );
	logMemoryUse( "Total", _mm_add_epi64( memModel, memContext ) );
	return S_OK;
}

HRESULT COMLIGHTCALL ContextImpl::timingsReset()
{
	profiler.reset();
	return S_OK;
}

HRESULT COMLIGHTCALL ContextImpl::getResults( eResultFlags flags, iTranscribeResult** pp ) const noexcept
{
	if( nullptr == pp )
		return E_POINTER;

	if( flags & eResultFlags::NewObject )
	{
		ComLight::CComPtr<ComLight::Object<TranscribeResult>> obj;
		CHECK( ComLight::Object<TranscribeResult>::create( obj ) );
		CHECK( makeResults( flags, *obj, true ) );
		obj.detach( pp );
		return S_OK;
	}
	else
	{
		CHECK( makeResults( flags, results, false ) );
		iTranscribeResult* res = &results;
		res->AddRef();
		*pp = res;
		return S_OK;
	}
}

inline int64_t scaleTime( int64_t wisperTicks )
{
	return MFllMulDiv( wisperTicks, 10'000'000, 100, 0 );
}

HRESULT COMLIGHTCALL ContextImpl::makeResults( eResultFlags flags, TranscribeResult& res, bool moveStrings ) const noexcept
{
	const size_t segments = result_all.size();
	// Resize both vectors
	try
	{
		res.segments.resize( segments );
		if( flags & eResultFlags::Tokens )
		{
			size_t tc = 0;
			for( const auto& s : result_all )
				tc += s.tokens.size();
			res.tokens.resize( tc );
		}
		else
			res.tokens.clear();

		res.segmentsText.clear();
		if( moveStrings )
			res.segmentsText.resize( segments );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	const Whisper::Vocabulary& vocab = model.shared->vocab;
	const Vocabulary::id tokenEot = vocab.token_eot;

	size_t tokensSoFar = 0;
	for( size_t i = 0; i < segments; i++ )
	{
		sSegment& rdi = res.segments[ i ];
		const auto& rsi = result_all[ i ];

		if( moveStrings )
		{
			res.segmentsText[ i ].swap( rsi.text );
			rdi.text = res.segmentsText[ i ].c_str();
		}
		else
			rdi.text = rsi.text.c_str();

		if( flags & eResultFlags::Timestamps )
		{
			// Offset the time relative to the start of the media
			rdi.time.begin = scaleTime( rsi.t0 ) + mediaTimeOffset;
			rdi.time.end = scaleTime( rsi.t1 ) + mediaTimeOffset;
		}
		else
			store16( &rdi.time, _mm_setzero_si128() );

		rdi.firstToken = (uint32_t)tokensSoFar;
		const size_t tc = rsi.tokens.size();
		rdi.countTokens = (uint32_t)tc;

		if( flags & eResultFlags::Tokens )
		{
			for( size_t i = 0; i < tc; i++ )
			{
				sToken& rdi = res.tokens[ tokensSoFar + i ];
				const auto& src = rsi.tokens[ i ];
				rdi.text = vocab.string( src.id );

				if( flags & eResultFlags::Timestamps )
				{
					// Offset the time relative to the start of the media
					rdi.time.begin = scaleTime( src.t0 ) + mediaTimeOffset;
					rdi.time.end = scaleTime( src.t1 ) + mediaTimeOffset;
				}
				else
					store16( &rdi.time, _mm_setzero_si128() );

				// Copy 4 floats with unaligned load and store instructions
				_mm_storeu_ps( &rdi.probability, _mm_loadu_ps( &src.p ) );

				rdi.id = src.id;

				uint32_t flags = 0;
				if( src.id >= tokenEot )
					flags |= (uint32_t)eTokenFlags::Special;
				rdi.flags = (eTokenFlags)flags;
			}
		}
		tokensSoFar += tc;
	}
	return S_OK;
}

int ContextImpl::wrapSegment( int max_len )
{
	// whisper_wrap_segment
	auto segment = result_all.back();
	int res = 1;
	int acc = 0;
	std::string text;
	const Whisper::Vocabulary& vocab = model.shared->vocab;
	const int tokenEot = vocab.token_eot;

	for( int i = 0; i < (int)segment.tokens.size(); i++ )
	{
		const auto& token = segment.tokens[ i ];
		if( token.id >= tokenEot )
			continue;

		const char* txt = vocab.string( token.id );
		const int cur = (int)strlen( txt );

		if( acc + cur > max_len && i > 0 )
		{
			// split here
			result_all.back().text = std::move( text );
			result_all.back().t1 = token.t0;
			result_all.back().tokens.resize( i );

			result_all.push_back( {} );
			result_all.back().t0 = token.t0;
			result_all.back().t1 = segment.t1;

			// add tokens [i, end] to the new segment
			result_all.back().tokens.insert( result_all.back().tokens.end(), segment.tokens.begin() + i, segment.tokens.end() );

			acc = 0;
			text = "";

			segment = result_all.back();
			i = -1;

			res++;
		}
		else
		{
			acc += cur;
			text += txt;
		}
	}

	result_all.back().text = std::move( text );
	return res;
}

HRESULT COMLIGHTCALL ContextImpl::runFull( const sFullParams& params, const iAudioBuffer* buffer )
{
#if SAVE_DEBUG_TRACE
	Tracing::vector( "runFull.pcm.in", buffer->getPcmMono(), buffer->countSamples() );
#endif
	CHECK( buffer->getTime( mediaTimeOffset ) );

	auto profCompleteCpu = profiler.cpuBlock( eCpuBlock::RunComplete );
	{
		auto p = profiler.cpuBlock( eCpuBlock::Spectrogram );
		CHECK( spectrogram.pcmToMel( buffer, model.shared->filters, params.cpuThreads ) );
	}

	if( params.flag( eFullParamsFlags::TokenTimestamps ) )
	{
		t_beg = 0;
		t_last = 0;
		tid_last = 0;
		computeSignalEnergy( energy, buffer, 32 );
	}

	try
	{
		sProgressSink progressSink{ nullptr, nullptr };
		return runFullImpl( params, progressSink, spectrogram );
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}

HRESULT COMLIGHTCALL ContextImpl::runStreamed( const sFullParams& params, const sProgressSink& progress, const iAudioReader* reader )
{
	if( params.flag( eFullParamsFlags::TokenTimestamps ) )
	{
		logError( u8"eFullParamsFlags.TokenTimestamps flag is not supported in streaming mode" );
		return E_NOTIMPL;
	}

	mediaTimeOffset = 0;
	auto profCompleteCpu = profiler.cpuBlock( eCpuBlock::RunComplete );

	try
	{
		if( params.cpuThreads > 1 )
		{
			MelStreamerThread mel{ model.shared->filters, profiler, reader, params.cpuThreads };
			return runFullImpl( params, progress, mel );
		}
		else
		{
			MelStreamerSimple mel{ model.shared->filters, profiler, reader };
			return runFullImpl( params, progress, mel );
		}
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}