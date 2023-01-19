#pragma once
#include <stdint.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>
#include "../Whisper/audioConstants.h"

namespace Whisper
{
	HRESULT createMediaType( bool stereo, IMFMediaType** pp );

	HRESULT getStreamDuration( IMFSourceReader* reader, int64_t& duration );

	HRESULT validateCurrentMediaType( IMFSourceReader* reader, uint32_t expectedChannels );

	struct iAudioReader;
	void setPreciseSamplesCount( const iAudioReader* ar, int64_t count );
}