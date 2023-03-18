#pragma warning disable CS0649  // Field is never assigned to

// Missing XML comment for publicly visible type or member
// TODO: remove this line and document them.
#pragma warning disable CS1591

using System;
using System.Runtime.InteropServices;

namespace Whisper.Internals
{
	/// <summary>This callback is called on each new segment</summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	delegate int pfnNewSegment( IntPtr ctx, int countNew, IntPtr userData );

	/// <summary>The callback is called before every encoder run. If it returns S_FALSE, the processing is aborted.</summary>
	[UnmanagedFunctionPointer( CallingConvention.Cdecl )]
	delegate int pfnEncoderBegin( IntPtr ctx, IntPtr userData );

	/// <summary>Transcribe parameters</summary>
	public struct sFullParams
	{
		internal Parameters publicParams;
		// The rest of these parameters are not exposed to the user-friendly public API of this DLL

		internal IntPtr prompt_tokens;
		internal int prompt_n_tokens;

		/// <summary>This callback is called on each new segment</summary>
		[MarshalAs( UnmanagedType.FunctionPtr )]
		internal pfnNewSegment newSegmentCallback;
		/// <summary>Parameter for the above, not needed in C#</summary>
		internal IntPtr newSegmentCallbackData;

		/// <summary>The callback is called before every encoder run. If it returns false, the processing is aborted</summary>
		[MarshalAs( UnmanagedType.FunctionPtr )]
		internal pfnEncoderBegin encoderBeginCallback;
		/// <summary>Parameter for the above, not needed in C#</summary>
		internal IntPtr encoderBeginCallbackData;
	}
}