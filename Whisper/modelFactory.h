#pragma once
#include "API/sLoadModelCallbacks.h"
#include "API/sModelSetup.h"

namespace Whisper
{
	struct iModel;

	HRESULT __stdcall loadGpuModel( const wchar_t* path, const sModelSetup& setup, const sLoadModelCallbacks* callbacks, iModel** pp );

	HRESULT __stdcall loadReferenceCpuModel( const wchar_t* path, iModel** pp );
}