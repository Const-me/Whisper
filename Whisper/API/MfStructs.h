#pragma once

namespace Whisper
{
	struct sCaptureDevice
	{
		// The display name is suitable for showing to the user, but might not be unique.
		const wchar_t* displayName;

		// Endpoint ID for an audio capture device
		// It uniquely identifies the device on the system, but is not a readable string.
		const wchar_t* endpoint;
	};

	using pfnFoundCaptureDevices = HRESULT( __stdcall* )( int len, const sCaptureDevice* buffer, void* pv );

	// Flags for the audio capture
	enum struct eCaptureFlags : uint32_t
	{
		// When the capture device supports stereo, keep stereo PCM samples in addition to mono
		Stereo = 1,
	};

	// Parameters for audio capture
	struct sCaptureParams
	{
		float minDuration = 2.0f;
		float maxDuration = 3.0f;
		float dropStartSilence = 0.25f;
		float pauseDuration = 0.333f;
		// Flags for the audio capture
		uint32_t flags = 0;
	};

	enum struct eCaptureStatus : uint8_t
	{
		Listening = 1,
		Voice = 2,
		Transcribing = 4,
		Stalled = 0x80,
	};

	// Return S_OK to continue, or S_FALSE to stop the capture session
	using pfnShouldCancel = HRESULT( __stdcall* )( void* pv ) noexcept;

	using pfnCaptureStatus = HRESULT( __stdcall* )( void* pv, eCaptureStatus status ) noexcept;

	struct sCaptureCallbacks
	{
		pfnShouldCancel shouldCancel;
		pfnCaptureStatus captureStatus;
		void* pv;
	};
}