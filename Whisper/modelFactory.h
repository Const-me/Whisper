#pragma once
#include "API/sLoadModelCallbacks.h"

namespace Whisper
{
	struct iModel;

	HRESULT __stdcall loadGpuModel( const wchar_t* path, bool hybrid, uint32_t flags, const sLoadModelCallbacks* callbacks, iModel** pp );

	HRESULT __stdcall loadReferenceCpuModel( const wchar_t* path, iModel** pp );
}