#pragma once

namespace Whisper
{
	using pfnLoadProgress = HRESULT( __stdcall* )( double val, void* pv ) noexcept;
	using pfnCancel = HRESULT( __stdcall* )( void* pv ) noexcept;
	
	struct sLoadModelCallbacks
	{
		pfnLoadProgress progress;
		pfnCancel cancel;
		void* pv;
	};
}