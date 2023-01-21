#include "stdafx.h"
#include "DelayExecution.h"

namespace
{
	constexpr bool useHighRezTimer = false;

	constexpr int64_t sleepMicroseconds = 200;

	inline HRESULT sleepImpl( HANDLE timer )
	{
		constexpr int64_t sleepTicks = sleepMicroseconds * 10;

		LARGE_INTEGER li;
		// Negative values indicate relative time
		li.QuadPart = -sleepTicks;
		if( !SetWaitableTimerEx( timer, &li, 0, nullptr, nullptr, nullptr, 0 ) )
			return getLastHr();
		const DWORD res = WaitForSingleObject( timer, 50 );
		if( res == WAIT_OBJECT_0 )
			return S_OK;
		if( res == WAIT_FAILED )
			return getLastHr();
		return E_FAIL;
	}
}

void DelayExecution::sleepOnTheTimer( const DelayExecution& delay )
{
	HRESULT hr = sleepImpl( delay.timer );
	if( SUCCEEDED( hr ) )
		return;
	logWarningHr( hr, u8"DelayExecution.sleepOnTheTimer" );
}

void DelayExecution::spinWait( const DelayExecution& )
{
	for( size_t i = 0; i < 1024; i++ )
		_mm_pause();
}

void DelayExecution::sleep( const DelayExecution& )
{
	Sleep( 0 );
}

DelayExecution::DelayExecution()
{
	if constexpr( useHighRezTimer )
	{
		constexpr DWORD flags = CREATE_WAITABLE_TIMER_HIGH_RESOLUTION;
		HANDLE h = CreateWaitableTimerEx( nullptr, nullptr, flags, TIMER_ALL_ACCESS );
		if( nullptr != h )
		{
			timer.Attach( h );
			pfn = &sleepOnTheTimer;
			return;
		}

		const HRESULT hr = getLastHr();
		logWarningHr( hr, u8"CreateWaitableTimerEx" );
	}

	pfn = &spinWait;
	// pfn = &sleep;
}