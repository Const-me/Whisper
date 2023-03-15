#pragma warning disable CS0649 // Field is never assigned to
using System;
using System.Runtime.InteropServices;

namespace Whisper.Internal
{
	/// <summary>Identifiers for an audio capture device</summary>
	public struct sCaptureDevice
	{
		readonly IntPtr m_displayName;
		/// <summary>The display name is suitable for showing to the user, but might not be unique.</summary>
		public string displayName => Marshal.PtrToStringUni( m_displayName );

		readonly IntPtr m_endpoint;
		/// <summary>Endpoint ID for an audio capture device.<br/>
		/// It uniquely identifies the device on the system, but is not a readable string.</summary>
		public string endpoint => Marshal.PtrToStringUni( m_endpoint );
	}

	/// <summary>Function pointer to consume a list of audio capture device IDs</summary>
	[UnmanagedFunctionPointer( CallingConvention.StdCall )]
	public delegate int pfnFoundCaptureDevices( int len, [In, MarshalAs( UnmanagedType.LPArray, SizeParamIndex = 0 )] sCaptureDevice[] arr, IntPtr pv );
}