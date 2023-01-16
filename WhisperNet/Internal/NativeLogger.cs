using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Whisper.Internal
{
	/// <summary>Utility class to supply logging function pointer to the C++ library,<br/>
	/// and provide custom calling conventions to ComLight runtime to convert error messages printed in C++ into .NET exception messages</summary>
	public static class NativeLogger
	{
		internal static void startup() { }

		static NativeLogger()
		{
			sink = logSink;
			sLoggerSetup setup = default;
			setup.sink = sink;
			setup.level = eLogLevel.Warning;
			Library.setupLogger( ref setup );
		}

		internal static void setup( eLogLevel lvl, eLoggerFlags flags, pfnLogMessage? pfn )
		{
			logMessage = pfn;

			sLoggerSetup setup = default;
			setup.sink = sink;
			setup.level = lvl;
			setup.flags = flags;
			Library.setupLogger( ref setup );
		}

		// This field is here to protect the function pointer from being collected by the GC
		static readonly pfnLoggerSink sink;

		static void logSink( IntPtr context, eLogLevel lvl, string message )
		{
			if( lvl == eLogLevel.Error )
				state.setText( message );
			logMessage?.Invoke( lvl, message );
		}

		sealed class ThreadState
		{
			string? errorText = null;
			ExceptionDispatchInfo? dispatchInfo = null;

			public void setText( string text ) => errorText = text;
			public void capture( Exception ex ) => dispatchInfo = ExceptionDispatchInfo.Capture( ex );

			public void clear()
			{
				errorText = null;
				dispatchInfo = null;
			}

			public void Deconstruct( out string? text, out ExceptionDispatchInfo? edi )
			{
				text = errorText;
				edi = dispatchInfo;
				errorText = null;
				dispatchInfo = null;
			}
		}

		[ThreadStatic]
		static ThreadState state = new ThreadState();

		internal static void captureException( Exception ex ) =>
			state.capture( ex );

		static pfnLogMessage? logMessage = null;

		/// <summary>Called internally by ComLight runtime</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static void prologue()
		{
			// https://stackoverflow.com/a/2043505/126995
			if( null != state )
				state.clear();
			else
				createState();
		}

		[MethodImpl( MethodImplOptions.NoInlining )]
		static void createState()
		{
			state = new ThreadState();
		}

		/// <summary>Epilogue implementation for unsuccessful status codes</summary>
		[MethodImpl( MethodImplOptions.NoInlining )]
		static void throwException( int hr )
		{
			// Move state from the thread local object into local variables, and clear that object
			(string? text, ExceptionDispatchInfo? edi) = state;

			if( null != edi && edi.SourceException.HResult == hr )
			{
				// The error comes from a callback, and we have original context of that exception.
				// Re-throw the original exception.
				// This uses the original error message, and even correctly deals with the stack trace.
				edi.Throw();
			}

			if( null != text )
			{
				// C++ code has printed an error on the current thread, between prologue and epilogue.
				// Use that text for the exception message.
				Exception? ex = Marshal.GetExceptionForHR( hr );
				throw new ApplicationException( text, ex );
			}

			// We don’t have any additional info about the exception.
			// Throw an exception from just the HRESULT code.
			Marshal.ThrowExceptionForHR( hr );
		}

		/// <summary>Called internally by ComLight runtime</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static void throwForHR( int hr )
		{
			if( hr >= 0 )
				return; // SUCCEEDED
			throwException( hr );
		}

		/// <summary>Called internally by ComLight runtime</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static bool throwAndReturnBool( int hr )
		{
			if( hr >= 0 )
				return 0 == hr;
			throwException( hr );
			return false;
		}
	}
}