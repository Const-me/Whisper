#pragma warning disable CS0649  // Field is never assigned to
using System;
using System.Runtime.InteropServices;

namespace Whisper.Internal
{
	/// <summary>A callback to get notified about the progress</summary>
	[UnmanagedFunctionPointer( CallingConvention.StdCall )]
	delegate int pfnReportProgress( double value, IntPtr context, IntPtr pv );

	/// <summary>C structure with a progress reporting function pointer</summary>
	public struct sProgressSink
	{
		/// <summary>A callback to get notified about the progress</summary>
		[MarshalAs( UnmanagedType.FunctionPtr )]
		internal pfnReportProgress pfn;

		/// <summary>Last parameter to the callback</summary>
		internal IntPtr pv;
	}
}