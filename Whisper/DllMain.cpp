#include "stdafx.h"

BOOL __stdcall DllMain( HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpvReserved )
{
	// Perform actions based on the reason for calling.
	switch( fdwReason )
	{
	case DLL_PROCESS_ATTACH:
		// Initialize once for each new process. Return FALSE to fail DLL load.
		DisableThreadLibraryCalls( (HMODULE)hinstDLL );
		break;
	case DLL_THREAD_ATTACH:
		// Do thread-specific initialization.
		break;
	case DLL_THREAD_DETACH:
		// Do thread-specific cleanup.
		break;
	case DLL_PROCESS_DETACH:
		if( lpvReserved != nullptr )
		{
			break; // do not do cleanup if process termination scenario
		}
		// Perform any necessary cleanup
		break;
	}
	return TRUE;  // Successful DLL_PROCESS_ATTACH.
}