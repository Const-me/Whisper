using System.Runtime.InteropServices;

namespace Whisper.Internal
{
	/// <summary>Function pointer to report model loading progress</summary>
	[UnmanagedFunctionPointer( CallingConvention.StdCall )]
	delegate int pfnLoadProgress( double progress, IntPtr pv );

	/// <summary>Function pointer to implement cooperative cancellation</summary>
	[UnmanagedFunctionPointer( CallingConvention.StdCall )]
	delegate int pfnCancel( IntPtr pv );

	/// <summary>Callback functions for loading models</summary>
	public struct sLoadModelCallbacks
	{
		/// <summary>Function pointer to report model loading progress</summary>
		[MarshalAs( UnmanagedType.FunctionPtr )]
		pfnLoadProgress? progress;

		/// <summary>Function pointer to implement cooperative cancellation</summary>
		[MarshalAs( UnmanagedType.FunctionPtr )]
		pfnCancel? cancel;

		// Not needed in C#, delegates can capture things
		IntPtr pv;

		/// <summary>Wrap idiomatic C# things into these low-level C callbacks</summary>
		internal sLoadModelCallbacks( CancellationToken cancelToken, Action<double>? pfnProgress )
		{
			if( cancelToken != CancellationToken.None )
			{
				cancel = delegate ( IntPtr pv )
				{
					if( cancelToken.IsCancellationRequested )
						return 1;   // S_FALSE
					return 0;   // S_OK
				};
			}
			else
				cancel = null;

			if( null != pfnProgress )
			{
				progress = delegate ( double val, IntPtr pv )
				{
					try
					{
						pfnProgress( val );
						return 0;   // S_OK
					}
					catch( Exception ex )
					{
						NativeLogger.captureException( ex );
						return ex.HResult;
					}
				};
			}
			else
				progress = null;

			pv = IntPtr.Zero;
		}
	}
}