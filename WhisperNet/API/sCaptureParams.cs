using System;

namespace Whisper
{
	/// <summary>Flags for the audio capture</summary>
	[Flags]
	public enum eCaptureFlags: uint
	{
		/// <summary>No special flags</summary>
		None = 0,
		/// <summary>When the capture device supports stereo, keep stereo PCM samples in addition to mono</summary>
		Stereo = 1,
	}

	/// <summary>Parameters for audio capture</summary>
	public struct sCaptureParams
	{
		/// <summary>Minimum transcribe duration in seconds</summary>
		public float minDuration;
		/// <summary>Maximum transcribe duration in seconds</summary>
		public float maxDuration;
		/// <summary></summary>
		public float dropStartSilence;
		/// <summary></summary>
		public float pauseDuration;
		/// <summary>Flags for the audio capture</summary>
		public eCaptureFlags flags;

		/// <summary>Initialize the structure with some reasonable default values</summary>
		public sCaptureParams( bool unused )
		{
			minDuration = 7.0f;         // 7 seconds
			maxDuration = 11.0f;        // 11 seconds
			dropStartSilence = 0.25f;   // 250 ms
			pauseDuration = 0.333f;     // 333 ms
			flags = eCaptureFlags.None;
		}
	}
}