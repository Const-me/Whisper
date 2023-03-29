#pragma once
#include "../API/iMediaFoundation.cl.h"

namespace Whisper
{
	HRESULT COMLIGHTCALL loadAudioFile( LPCTSTR path, bool stereo, iAudioBuffer** pp );
	HRESULT COMLIGHTCALL loadAudioMemoryFile(const void* Buffer, uint64_t len, bool stereo, iAudioBuffer** pp);
}