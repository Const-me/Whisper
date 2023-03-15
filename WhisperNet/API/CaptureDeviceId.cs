using System;
using Whisper.Internal;

namespace Whisper
{
	/// <summary>Identifiers for an audio capture device</summary>
	public struct CaptureDeviceId
	{
		/// <summary>The display name is suitable for showing to the user, but might not be unique.</summary>
		public string displayName;

		/// <summary>Endpoint ID for an audio capture device.<br/>
		/// It uniquely identifies the device on the system, but is not a readable string.</summary>
		public string endpoint;

		internal CaptureDeviceId( in sCaptureDevice rsi )
		{
			displayName = rsi.displayName ?? "<display name unavailable>";
			endpoint = rsi.endpoint ?? throw new ApplicationException( "The device has no endpoint ID" );
		}

		/// <summary>Returns a String which represents the object instance</summary>
		public override string ToString() => $"Capture device: \"{displayName}\"";
	}
}