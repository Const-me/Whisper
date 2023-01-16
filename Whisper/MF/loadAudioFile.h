#pragma once
#include "../API/iMediaFoundation.cl.h"

namespace Whisper
{
	HRESULT COMLIGHTCALL loadAudioFile( LPCTSTR path, bool stereo, iAudioBuffer** pp );
}