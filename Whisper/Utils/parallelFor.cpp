#include "stdafx.h"
#include "parallelFor.h"

namespace
{
	class alignas( 64 ) ParallelForContext
	{
		volatile long threadIndex;
		volatile HRESULT status;

		alignas( 64 ) void* const context;
		const Whisper::pfnParallelForCallback pfn;

		static void __stdcall callbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID pv, PTP_WORK Work );

	public:

		ParallelForContext( void* ctx, Whisper::pfnParallelForCallback pfn );

		PTP_WORK createWork();

		HRESULT getStatus() const;
	};

	ParallelForContext::ParallelForContext( void* ctx, Whisper::pfnParallelForCallback callback ) :
		threadIndex( 1 ),
		status( S_FALSE ),
		context( ctx ),
		pfn( callback )
	{ }

	PTP_WORK ParallelForContext::createWork()
	{
		return CreateThreadpoolWork( &callbackStatic, this, nullptr );
	}

	void __stdcall ParallelForContext::callbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID pv, PTP_WORK Work )
	{
		ParallelForContext& context = *(ParallelForContext*)pv;
		int ith = InterlockedIncrement( &context.threadIndex );
		ith--;
		const HRESULT hr = context.pfn( ith, context.context );
		if( SUCCEEDED( hr ) )
			return;
		InterlockedCompareExchange( &context.status, hr, S_FALSE );
	}

	HRESULT ParallelForContext::getStatus() const
	{
		const HRESULT hr = status;
		if( SUCCEEDED( hr ) )
			return S_OK;
		return hr;
	}
}

namespace Whisper
{
	HRESULT parallelFor( pfnParallelForCallback pfn, int threadsCount, void* ctx )
	{
		if( threadsCount < 1 )
			return E_BOUNDS;
		if( threadsCount == 1 )
			return pfn( 0, ctx );

		ParallelForContext context{ ctx, pfn };

		PTP_WORK const pw = context.createWork();
		if( nullptr == pw )
			return getLastHr();

		for( int i = 1; i < threadsCount; i++ )
			SubmitThreadpoolWork( pw );

		const HRESULT hr0 = pfn( 0, ctx );

		WaitForThreadpoolWorkCallbacks( pw, FALSE );
		CloseThreadpoolWork( pw );

		if( FAILED( hr0 ) )
			return hr0;
		return context.getStatus();
	}
}

using namespace Whisper;

ThreadPoolWork::~ThreadPoolWork()
{
	if( nullptr != work )
	{
		CloseThreadpoolWork( work );
		work = nullptr;
	}
}

HRESULT ThreadPoolWork::create()
{
	if( nullptr == work )
	{
		work = CreateThreadpoolWork( &callbackStatic, this, nullptr );
		if( nullptr != work )
			return S_OK;
		return getLastHr();
	}
	return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );
}

HRESULT ThreadPoolWork::parallelFor( int threadsCount ) noexcept
{
	if( nullptr != work )
	{
		if( threadsCount <= 1 )
			return threadPoolCallback( 0 );

		threadIndex = 1;
		status = S_FALSE;
		for( int i = 1; i < threadsCount; i++ )
			SubmitThreadpoolWork( work );

		const HRESULT hr0 = threadPoolCallback( 0 );

		WaitForThreadpoolWorkCallbacks( work, FALSE );

		if( FAILED( hr0 ) )
			return hr0;
		if( SUCCEEDED( status ) )
			return S_OK;
		return status;
	}

	return OLE_E_BLANK;
}

void __stdcall ThreadPoolWork::callbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID pv, PTP_WORK Work )
{
	ThreadPoolWork* tpw = (ThreadPoolWork*)pv;
	int ith = InterlockedIncrement( &tpw->threadIndex );
	ith--;
	const HRESULT hr = tpw->threadPoolCallback( ith );
	if( SUCCEEDED( hr ) )
		return;
	InterlockedCompareExchange( &tpw->status, hr, S_FALSE );
}