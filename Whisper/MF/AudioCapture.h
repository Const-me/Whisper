#pragma once
#include "../API/MfStructs.h"

namespace Whisper
{
	struct iAudioCapture;
	struct iMediaFoundation;

	HRESULT __stdcall captureDeviceList( pfnFoundCaptureDevices pfn, void* pv );

	HRESULT __stdcall captureOpen( iMediaFoundation* owner, const wchar_t* endpoint, const sCaptureParams& captureParams, iAudioCapture** pp ) noexcept;
}