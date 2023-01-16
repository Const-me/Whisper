#include "stdafx.h"
#include "mfStartup.h"
#include <atlbase.h>
#include <mfapi.h>
#pragma comment(lib, "Mfplat.lib")

namespace
{
	struct sCoInitStatus
	{
		// Possible state:
		// -1 is the initial state, coInitialize never called
		// S_OK - CoInitializeEx succeeded, in this state the counter tracks the count of coInitialize() for the current thread
		// S_FALSE - CoInitializeEx failed with RPC_E_CHANGED_MODE, or did nothing because already initialized for the current thread
		// Error status - CoInitializeEx failed for some other reason
		HRESULT code = -1;
		uint32_t counter = 0;
	};
	thread_local sCoInitStatus coInitStatus;

	static HRESULT coInitialize()
	{
		sCoInitStatus& cis = coInitStatus;
		HRESULT hr = cis.code;
		if( SUCCEEDED( hr ) )
		{
			if( S_OK == hr )
				cis.counter++;
			return S_FALSE;
		}

		if( hr == HRESULT( -1 ) )
		{
			hr = CoInitializeEx( nullptr, COINIT_MULTITHREADED );
			if( S_OK == hr )
			{
				cis.counter = 1;
				return cis.code = S_OK;
			}
			if( S_FALSE == hr || RPC_E_CHANGED_MODE == hr )
			{
				return cis.code = S_FALSE;
			}
			cis.code = hr;
			return hr;
		}
		
		return hr;
	}

	static void coUninitialize()
	{
		sCoInitStatus& cis = coInitStatus;
		if( cis.code == S_OK )
		{
			assert( cis.counter > 0 );
			cis.counter--;
			if( 0 == cis.counter )
				CoUninitialize();
		}
	}

	static CComAutoCriticalSection s_lock;
#define LOCK() CComCritSecLock<CComAutoCriticalSection> lock{ s_lock }
	static uint32_t mfStartupCounter = 0;

	constexpr uint8_t FlagCOM = 1;
	constexpr uint8_t FlagMF = 0x10;
}

using namespace Whisper;

MfStartupRaii::~MfStartupRaii()
{
	if( 0 != ( successFlags & FlagMF ) )
	{
		LOCK();
		assert( mfStartupCounter > 0 );
		mfStartupCounter--;
		if( mfStartupCounter > 0 )
			return;
		MFShutdown();
		successFlags &= ~FlagMF;
	}
	
	if( 0 != ( successFlags & FlagCOM ) )
	{
		coUninitialize();
		successFlags &= ~FlagCOM;
	}
}

HRESULT MfStartupRaii::startup()
{
	if( 0 != ( successFlags & FlagMF ) )
		return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );

	HRESULT hr = coInitialize();
	CHECK( hr );
	if( hr == S_OK )
		successFlags |= FlagCOM;

	LOCK();

	if( 0 == mfStartupCounter )
	{
		HRESULT hr = MFStartup( MF_VERSION, MFSTARTUP_LITE );
		if( SUCCEEDED( hr ) )
		{
			mfStartupCounter = 1;
			successFlags |= FlagMF;
			return S_OK;
		}

		if( 0 != ( successFlags & FlagCOM ) )
		{
			coUninitialize();
			successFlags &= ~FlagCOM;
		}
		return hr;
	}
	else
	{
		mfStartupCounter++;
		successFlags |= FlagMF;
		return S_FALSE;
	}
}