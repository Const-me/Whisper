#include "stdafx.h"
#include "ParallelForRunner.h"
using namespace CpuCompute;

ParallelForRunner::ParallelForRunner( int threads ) :
	maxThreads( threads )
{
	if( maxThreads <= 1 )
	{
		threadBuffers.resize( 1 );
		return;
	}

	work = CreateThreadpoolWork( &workCallbackStatic, this, nullptr );
	if( nullptr == work )
		throw getLastHr();
	threadBuffers.resize( maxThreads );
}

HRESULT ParallelForRunner::setThreadsCount( int threads )
{
	maxThreads = threads;
	if( threads <= 1 )
	{
		threadBuffers.resize( 1 );
		return S_OK;
	}

	threadBuffers.resize( maxThreads );
	if( nullptr == work )
	{
		work = CreateThreadpoolWork( &workCallbackStatic, this, nullptr );
		if( nullptr == work )
			return getLastHr();
	}
	return S_OK;
}

ParallelForRunner::~ParallelForRunner()
{
	if( nullptr != work )
	{
		if( S_FALSE == status )
			WaitForThreadpoolWorkCallbacks( work, FALSE );
		CloseThreadpoolWork( work );
	}
}

namespace
{
	thread_local uint32_t currentThreadIndex = UINT_MAX;
}

void ParallelForRunner::runBatch( size_t ith ) noexcept
{
	currentThreadIndex = (uint32_t)ith;
	const size_t begin = ( ith * countItems ) / countThreads;
	const size_t end = ( ( ith + 1 ) * countItems ) / countThreads;

	HRESULT hr = E_UNEXPECTED;
	try
	{
		hr = computeRange->compute( begin, end );
	}
	catch( HRESULT code )
	{
		hr = code;
	}
	catch( const std::bad_alloc& )
	{
		hr = E_OUTOFMEMORY;
	}
	catch( const std::exception& )
	{
		hr = E_FAIL;
	}
	currentThreadIndex = UINT_MAX;
	if( SUCCEEDED( hr ) )
		return;
	InterlockedCompareExchange( &status, hr, S_FALSE );
}

void* ParallelForRunner::threadLocalBuffer( size_t cb )
{
	const uint32_t idx = currentThreadIndex;
	if( idx < threadBuffers.size() )
	{
		ThreadBuffer& tb = threadBuffers[ idx ];
		if( tb.cb >= cb )
		{
			// We already have large enough buffer for the current thread
			return tb.memory.pointer();
		}
		tb.memory.deallocate();
		check( tb.memory.allocate( cb ) );
		tb.cb = cb;
		return tb.memory.pointer();
	}
	if( idx != UINT_MAX )
		throw E_BOUNDS;
	else
	{
		logError( u8"threadLocalBuffer() method only works from inside a pool callback" );
		throw E_UNEXPECTED;
	}
}

void __stdcall ParallelForRunner::workCallbackStatic( PTP_CALLBACK_INSTANCE Instance, void* pv, PTP_WORK Work ) noexcept
{
	ParallelForRunner& context = *(ParallelForRunner*)pv;
	const size_t ith = (uint32_t)( InterlockedIncrement( &context.threadIndex ) );
	context.runBatch( ith );
}

HRESULT ParallelForRunner::parallelFor( iComputeRange& compute, size_t length, size_t minBatch )
{
	if( maxThreads <= 1 )
	{
		currentThreadIndex = 0;
		const HRESULT hr1 = compute.compute( 0, length );
		currentThreadIndex = UINT_MAX;
		return hr1;
	}
	assert( minBatch > 0 );

	size_t nth = length / minBatch;
	nth = std::min( nth, (size_t)(uint32_t)maxThreads );

	computeRange = &compute;
	countItems = length;
	countThreads = nth;
	threadIndex = 0;
	status = S_FALSE;

	for( size_t i = 1; i < nth; i++ )
		SubmitThreadpoolWork( work );
	runBatch( 0 );

	if( nth > 1 )
		WaitForThreadpoolWorkCallbacks( work, FALSE );

	computeRange = nullptr;
	const HRESULT hr = status;
	status = S_OK;
	if( SUCCEEDED( hr ) )
		return S_OK;

	return hr;
}