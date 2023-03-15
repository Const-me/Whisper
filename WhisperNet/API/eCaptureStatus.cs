using System;

namespace Whisper
{
	/// <summary>Status of the voice capture</summary>
	[Flags]
	public enum eCaptureStatus: byte
	{
		/// <summary>Doing nothing</summary>
		None = 0,
		/// <summary>Capturing the audio</summary>
		Listening = 1,
		/// <summary>A voice is detected in the captured audio, recording</summary>
		Voice = 2,
		/// <summary>Transcribing a recorded piece of the audio</summary>
		Transcribing = 4,
		/// <summary>The computer is unable to transcribe the audio quickly enough,<br/>
		/// and the capture is dropping the incoming audio samples.</summary>
		Stalled = 0x80,
	}
}