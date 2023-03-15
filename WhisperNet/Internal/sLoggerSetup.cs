using System;
using System.Runtime.InteropServices;

namespace Whisper.Internal
{
	[UnmanagedFunctionPointer( CallingConvention.StdCall )]
	delegate void pfnLoggerSink( IntPtr context, eLogLevel lvl, [MarshalAs( UnmanagedType.LPUTF8Str )] string message );

	struct sLoggerSetup
	{
		[MarshalAs( UnmanagedType.FunctionPtr )]
		public pfnLoggerSink sink;
		IntPtr context;
		public eLogLevel level;
		public eLoggerFlags flags;
	}
}