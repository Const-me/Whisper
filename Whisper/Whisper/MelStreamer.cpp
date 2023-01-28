#include "stdafx.h"
#include "MelStreamer.h"
#include "../Utils/parallelFor.h"
using namespace Whisper;

MelStreamer::MelStreamer( const Filters& filters, ProfileCollection& prof, const iAudioReader* iar ) :
	reader( iar ),
	melContext( filters ),
	profiler( prof )
{ }

void MelStreamer::dropOldChunks( size_t off )
{
	const bool stereo = reader.outputsStereo();
	for( size_t i = streamStartOffset; i < off; i++ )
	{
		queuePcmMono.pop_front();
		queueMel.pop_front();
		if( stereo )
			queuePcmStereo.pop_front();
	}
	streamStartOffset = off;
}

HRESULT MelStreamer::ensurePcmChunks( size_t len )
{
	if( readerEof )
		return queuePcmMono.empty() ? E_EOF : S_FALSE;

	const bool loadStereo = reader.outputsStereo();

	const size_t neededChunks = len + FFT_SIZE / FFT_STEP;
	while( true )
	{
		if( queuePcmMono.size() >= neededChunks )
			return S_OK;

		PcmMonoChunk& mono = queuePcmMono.emplace_back();
		PcmStereoChunk* stereo = loadStereo ? &queuePcmStereo.emplace_back() : nullptr;
		HRESULT hr = reader.readChunk( mono, stereo );
		if( SUCCEEDED( hr ) )
			continue;

		queuePcmMono.pop_back();
		if( loadStereo )
			queuePcmStereo.pop_back();

		if( hr == E_EOF )
		{
			readerEof = true;
			return S_FALSE;
		}

		return hr;
	}
}

size_t MelStreamer::serializePcm( size_t startOffset )
{
	const ptrdiff_t chunks = (ptrdiff_t)queuePcmMono.size() - (ptrdiff_t)startOffset;
	assert( chunks > 0 );

	tempPcm.resize( chunks * FFT_STEP );
	float* rdi = tempPcm.data();

	for( auto it = queuePcmMono.begin() + startOffset; it != queuePcmMono.end(); it++ )
	{
		memcpy( rdi, it->mono.data(), FFT_STEP * 4 );
		rdi += FFT_STEP;
	}
	return chunks;
}

namespace
{
	__forceinline __m128 transpose4x80( __m128 vmax, const float* c0, const float* c1, const float* c2, const float* c3, float* rdi, size_t stride )
	{
		const float* const c0End = c0 + 80;
		for( ; c0 < c0End; c0 += 4, c1 += 4, c2 += 4, c3 += 4, rdi += stride * 4 )
		{
			__m128 r0 = _mm_loadu_ps( c0 );
			__m128 r1 = _mm_loadu_ps( c1 );
			__m128 r2 = _mm_loadu_ps( c2 );
			__m128 r3 = _mm_loadu_ps( c3 );

			__m128 ax01 = _mm_max_ps( r0, r1 );
			__m128 ax02 = _mm_max_ps( r2, r3 );
			__m128 ax = _mm_max_ps( ax01, ax02 );
			vmax = _mm_max_ps( vmax, ax );

			_MM_TRANSPOSE4_PS( r0, r1, r2, r3 );

			_mm_storeu_ps( rdi, r0 );
			_mm_storeu_ps( rdi + stride, r1 );
			_mm_storeu_ps( rdi + stride * 2, r2 );
			_mm_storeu_ps( rdi + stride * 3, r3 );
		}
		return vmax;
	}

	__forceinline __m128 transpose80( __m128 vmax, const float* c0, float* rdi, size_t stride )
	{
		const float* const c0End = c0 + 80;
		for( ; c0 < c0End; c0 += 4, rdi += stride * 4 )
		{
			__m128 r0 = _mm_loadu_ps( c0 );
			vmax = _mm_max_ps( vmax, r0 );

			_mm_store_ss( rdi, r0 );
			*(int*)( rdi + stride ) = _mm_extract_ps( r0, 1 );
			*(int*)( rdi + stride * 2 ) = _mm_extract_ps( r0, 2 );
			*(int*)( rdi + stride * 3 ) = _mm_extract_ps( r0, 3 );
		}
		return vmax;
	}

	__forceinline float horizontalMaximum( __m128 v )
	{
		v = _mm_max_ps( v, _mm_movehl_ps( v, v ) );
		v = _mm_max_ss( v, _mm_movehdup_ps( v ) );
		return _mm_cvtss_f32( v );
	}
}

void MelStreamer::makeTransposedBuffer( size_t off, size_t len )
{
	// Resize the output
	assert( len <= queueMel.size() );
	outputMel.resize( len * N_MEL );	// N_MEL = 80

	// First pass, copy transposed MEL data, and compute the maximum
	const size_t lengthAligned = ( len / 4 ) * 4;
	__m128 vMax = _mm_set1_ps( 1e-20f );
	float* rdi = outputMel.data();

	size_t i;
	for( i = 0; i < lengthAligned; i += 4, rdi += 4 )
	{
		vMax = transpose4x80( vMax,
			queueMel[ i ].data(),
			queueMel[ i + 1 ].data(),
			queueMel[ i + 2 ].data(),
			queueMel[ i + 3 ].data(),
			rdi, len );
	}
	for( ; i < len; i++, rdi++ )
		vMax = transpose80( vMax, queueMel[ i ].data(), rdi, len );

	// Second pass, clamping and normalization
	float mmax;
	const size_t bufferEnd = off + len;
	if( lastBufferEnd != bufferEnd )
	{
		// Store maximum value in this class, along with the end sample index
		mmax = horizontalMaximum( vMax );
		lastBufferEnd = bufferEnd;
		lastBufferMax = mmax;
	}
	else
	{
		// We're probably at the and of the stream, the caller asked for a smalled slice of the samples with the same end as the last time.
		// Discard the computed maximum value, and instead use the number stored in this class
		mmax = lastBufferMax;
	}

	mmax -= 8.0f;
	vMax = _mm_set1_ps( mmax );

	rdi = outputMel.data();
	float* const rdiEnd = rdi + outputMel.size();
	const __m128 add = _mm_set1_ps( 4 );
	const __m128 mul = _mm_set1_ps( 1.0f / 4.0f );
	for( ; rdi < rdiEnd; rdi += 4 )
	{
		__m128 v = _mm_loadu_ps( rdi );
		v = _mm_max_ps( v, vMax );
		v = _mm_add_ps( v, add );
		v = _mm_mul_ps( v, mul );
		_mm_storeu_ps( rdi, v );
	}
}

HRESULT MelStreamerSimple::makeBuffer( size_t off, size_t len, const float** buffer, size_t& stride ) noexcept
{
	if( off < streamStartOffset )
	{
		logError( u8"MelStreamer doesn't support backwards seeks" );
		return E_UNEXPECTED;
	}

	if( off > streamStartOffset )
	{
		// The model wants to advance forward, drop now irrelevant chunks of data
		dropOldChunks( off );
	}

	// Compute all these MEL chunks
	const size_t availableMel = queueMel.size();
	if( availableMel < len )
	{
		CHECK( ensurePcmChunks( len ) );

		const size_t pcmChunks = serializePcm( availableMel );
		const size_t missingMelChunks = len - availableMel;
		size_t i;
		const size_t loop1 = std::min( missingMelChunks, pcmChunks );
		{
			auto profilerBlock = profiler.cpuBlock( eCpuBlock::Spectrogram );
			for( i = 0; i < loop1; i++ )
			{
				// if( readerEof && i + 1 == loop1 ) __debugbreak();
				auto& arr = queueMel.emplace_back();
				const float* sourcePcm = tempPcm.data() + i * FFT_STEP;
				size_t availableChunks = pcmChunks - i;
				size_t availableFloats = availableChunks * FFT_STEP;
				melContext.fft( arr, sourcePcm, availableFloats );
			}
		}
		for( ; i < missingMelChunks; i++ )
		{
			assert( readerEof );
			auto& arr = queueMel.emplace_back();
			memset( arr.data(), 0, N_MEL * 4 );
		}
	}

	// Produce the result
	makeTransposedBuffer( off, len );
	stride = len;
	*buffer = outputMel.data();
	return S_OK;
}

MelStreamerThread::MelStreamerThread( const Filters& filters, ProfileCollection& profiler, const iAudioReader* iar, int countThreads ) :
	MelStreamer( filters, profiler, iar ),
	workerThreads( countThreads )
{
	if( workerThreads > 1 )
	{
		check( ThreadPoolWork::create() );
		melContextsWorkers.reserve( workerThreads - 1 );
		for( int i = 1; i < workerThreads; i++ )
			melContextsWorkers.emplace_back( filters );
	}

	InitializeConditionVariable( &wakeMain );
	InitializeConditionVariable( &wakeBackground );
	threadStatus = eThreadStatus::NotStarted;
	const HANDLE h = CreateThread( nullptr, 0, &threadProcStatic, this, 0, nullptr );
	if( nullptr == h )
		throw HRESULT_FROM_WIN32( GetLastError() );
	threadHandle.Attach( h );
}

using Lock = CComCritSecLock<CComAutoCriticalSection>;

constexpr ptrdiff_t prebufferChunks = 3000 * 2;
constexpr ptrdiff_t chunksPerWakeup = 512;
constexpr ptrdiff_t minChunksPerThread = 64;

HRESULT MelStreamerThread::threadMain()
{
	pendingChunks.reserve( chunksPerWakeup );

	EnterCriticalSection( &m_cs.m_sec );
	threadStatus = eThreadStatus::Working;

	while( true )
	{
		if( shuttingDown )
		{
			LeaveCriticalSection( &m_cs.m_sec );
			return S_FALSE;
		}

		// Count of available MEL chunks
		const ptrdiff_t availableMel = queueMel.size();
		if( availableMel >= prebufferChunks )
		{
			threadStatus = eThreadStatus::Idle;
			SleepConditionVariableCS( &wakeBackground, &m_cs.m_sec, INFINITE );
			threadStatus = eThreadStatus::Working;
			continue;
		}
		// Count of MEL chunks remaining in the whole stream
		// availableMel of them are already on the queue
		const ptrdiff_t remainingMel = (ptrdiff_t)getLength() - (ptrdiff_t)streamStartOffset;
		LeaveCriticalSection( &m_cs.m_sec );

		const ptrdiff_t missingChunks = prebufferChunks - availableMel;
		ptrdiff_t chunks = std::min( missingChunks, chunksPerWakeup );
		chunks = std::min( chunks, remainingMel - availableMel );
		if( chunks <= 0 )
			return S_OK; // This thread has produced all chunks of the stream

		CHECK( ensurePcmChunks( availableMel + chunks ) );
		const size_t pcmChunks = serializePcm( availableMel );
		if( 0 == pcmChunks )
			return S_OK;

		pendingChunks.clear();

		chunks = std::min( chunks, (ptrdiff_t)pcmChunks );
		{
			auto profilerBlock = profiler.cpuBlock( eCpuBlock::Spectrogram );

			if( this->workerThreads <= 1 || chunks < minChunksPerThread * 2 )
			{
				// Thread pool disabled with a setting, or not enough work for the thread pool
				for( ptrdiff_t i = 0; i < chunks; i++ )
				{
					MelChunk& arr = pendingChunks.emplace_back();
					const float* sourcePcm = tempPcm.data() + i * FFT_STEP;
					size_t availableChunks = pcmChunks - i;
					size_t availableFloats = availableChunks * FFT_STEP;
					melContext.fft( arr, sourcePcm, availableFloats );
				}
			}
			else
			{
				// Use thread pool for these FFTs
				pendingChunks.resize( chunks );
				int nth = (int)( ( chunks + minChunksPerThread - 1 ) / minChunksPerThread );
				nth = std::min( nth, this->workerThreads );
				assert( nth > 1 );
				this->fftChunks = (int)chunks;
				this->fftThreads = nth;
				CHECK( ThreadPoolWork::parallelFor( nth ) );
			}
		}

		EnterCriticalSection( &m_cs.m_sec );
		if( shuttingDown )
		{
			LeaveCriticalSection( &m_cs.m_sec );
			return S_FALSE;
		}

		for( const auto& a : pendingChunks )
			queueMel.push_back( a );

		LeaveCriticalSection( &m_cs.m_sec );

		WakeAllConditionVariable( &wakeMain );
		pendingChunks.clear();

		EnterCriticalSection( &m_cs.m_sec );
	}
}

HRESULT MelStreamerThread::threadPoolCallback( int ith ) noexcept
{
	SpectrogramContext& ctx = ( 0 != ith ) ? melContextsWorkers[ ith - 1 ] : melContext;

	// Figure out the slice of the chunks to generate in this thread
	const int nth = this->fftThreads;
	const int chunks = this->fftChunks;
	const int i0 = ( ith * chunks ) / nth;
	const int i1 = ( ( ith + 1 ) * chunks ) / nth;

	// Run these FFTs
	const size_t pcmChunks = tempPcm.size() / FFT_STEP;
	for( int i = i0; i < i1; i++ )
	{
		MelChunk& arr = pendingChunks[ i ];
		const float* sourcePcm = tempPcm.data() + i * FFT_STEP;
		size_t availableChunks = pcmChunks - i;
		size_t availableFloats = availableChunks * FFT_STEP;
		ctx.fft( arr, sourcePcm, availableFloats );
	}
	return S_OK;
}

HRESULT MelStreamerThread::run() noexcept
{
	HRESULT status;
	try
	{
		status = threadMain();
	}
	catch( HRESULT hr )
	{
		status = hr;
	}
	catch( const std::bad_alloc& )
	{
		status = E_OUTOFMEMORY;
	}
	catch( const std::exception& )
	{
		status = E_FAIL;
	}

	{
		Lock lk( m_cs );
		threadStatus = SUCCEEDED( status ) ? eThreadStatus::Completed : eThreadStatus::Failed;
	}

	// Especially when things fail, we want to wake the main thread up, so it's aware of the situation.
	WakeAllConditionVariable( &wakeMain );
	return status;
}

DWORD __stdcall MelStreamerThread::threadProcStatic( void* lpParameter )
{
	setCurrentThreadName( "Whisper.dll MEL Streamer Thread" );
	MelStreamerThread* p = (MelStreamerThread*)lpParameter;
	return (DWORD)p->run();
}

HRESULT MelStreamerThread::makeBuffer( size_t off, size_t len, const float** buffer, size_t& stride ) noexcept
{
	bool wakeThread = false;

	{
		Lock lock( m_cs );
		if( off < streamStartOffset )
		{
			logError( u8"MelStreamer doesn't support backwards seeks" );
			return E_UNEXPECTED;
		}

		if( off > streamStartOffset )
		{
			// The model wants to advance forward, drop now irrelevant chunks of data
			dropOldChunks( off );
			wakeThread = ( threadStatus == eThreadStatus::Working || threadStatus == eThreadStatus::Idle );
		}

		while( true )
		{
			const size_t availableMel = queueMel.size();
			if( availableMel >= len )
				break;

			const eThreadStatus ts = threadStatus;
			if( ts == eThreadStatus::Working || ts == eThreadStatus::Idle )
			{
				WakeAllConditionVariable( &wakeBackground );
				SleepConditionVariableCS( &wakeMain, &m_cs.m_sec, INFINITE );
				continue;
			}
			if( ts == eThreadStatus::Failed )
			{
				DWORD code;
				if( GetExitCodeThread( threadHandle, &code ) )
					return (HRESULT)code;
				else
					return HRESULT_FROM_WIN32( GetLastError() );
			}
			assert( ts == eThreadStatus::Completed );
			break;
		}

		if( queueMel.size() < len )
		{
			assert( readerEof || threadStatus == eThreadStatus::Failed );
			while( queueMel.size() < len )
			{
				auto& arr = queueMel.emplace_back();
				memset( arr.data(), 0, N_MEL * 4 );
			}
		}

		// Produce the result
		makeTransposedBuffer( off, len );

	}	// Unlock the critical section

	stride = len;
	*buffer = outputMel.data();
	if( wakeThread )
		WakeAllConditionVariable( &wakeBackground );
	return S_OK;
}

MelStreamerThread::~MelStreamerThread()
{
	if( !threadHandle )
		return;

	{
		Lock lock( m_cs );
		if( threadStatus != eThreadStatus::Working )
			return;
		shuttingDown = true;
	}

	DWORD res = WaitForSingleObject( threadHandle, 100 );
	if( res == WAIT_OBJECT_0 )
		return;
	// TODO: log a warning
}

HRESULT MelStreamer::copyStereoPcm( size_t offset, size_t length, std::vector<StereoSample>& buffer ) const
{
	if( queuePcmStereo.empty() )
		return OLE_E_BLANK;

	if( offset < streamStartOffset )
	{
		logError( u8"MelStreamer doesn't support backwards seek" );
		return E_UNEXPECTED;
	}

	// Offset relative to the first chunk on the queue
	const size_t off = offset - streamStartOffset;
	if( off >= queuePcmStereo.size() )
		return E_BOUNDS;

	// Resize the output buffer
	try
	{
		buffer.resize( length * FFT_STEP );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}
	StereoSample* rdi = buffer.data();

	// Copy PCM chunks from the queue
	const size_t lengthToCopy = std::min( length, queuePcmStereo.size() - off );
	for( size_t i = 0; i < lengthToCopy; i++, rdi += FFT_STEP )
	{
		const float* rsi = queuePcmStereo[ i + off ].stereo.data();
		memcpy( rdi, rsi, 8 * FFT_STEP );
	}
	// If needed, write zeros to the tail
	if( lengthToCopy == length )
		return S_OK;
	memset( rdi, 0, ( length - lengthToCopy ) * FFT_STEP );
	return S_OK;
}