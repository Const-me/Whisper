using ComLight;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using Whisper.Internal;

namespace Whisper
{
	/// <summary>Factory methods implemented by the C++ DLL</summary>
	public static class Library
	{
		static Library()
		{
			if( Environment.OSVersion.Platform != PlatformID.Win32NT )
				throw new ApplicationException( "This library requires Windows OS" );
			if( !Environment.Is64BitProcess )
				throw new ApplicationException( "This library only works in 64-bit processes" );
			if( RuntimeInformation.ProcessArchitecture != Architecture.X64 )
				throw new ApplicationException( "This library requires a processor with AMD64 instruction set" );
			if( !Sse41.IsSupported )
				throw new ApplicationException( "This library requires a CPU with SSE 4.1 support" );
			NativeLogger.startup();
		}

		const string dll = "Whisper.dll";

		[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = false )]
		internal static extern void setupLogger( [In] ref sLoggerSetup setup );

		[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
		static extern int loadModel( [MarshalAs( UnmanagedType.LPWStr )] string path, eModelImplementation impl,
			[In] ref sLoadModelCallbacks callbacks,
			[MarshalAs( UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof( Marshaler<iModel> ) )] out iModel model );

		/// <summary>Load Whisper model from GGML file on disk</summary>
		/// <remarks>Models are large, depending on user’s disk speed this might take a while, and this function blocks the calling thread.<br/>
		/// Consider <see cref="loadModelAsync" /> instead.</remarks>
		/// <seealso href="https://huggingface.co/datasets/ggerganov/whisper.cpp" />
		public static iModel loadModel( string path, eModelImplementation impl = eModelImplementation.GPU )
		{
			iModel model;
			sLoadModelCallbacks callbacks = default;
			NativeLogger.prologue();
			int hr = loadModel( path, impl, ref callbacks, out model );
			NativeLogger.throwForHR( hr );
			return model;
		}

		/// <summary>Load Whisper model on a background thread, with optional progress reporting and cancellation</summary>
		public static Task<iModel> loadModelAsync( string path, CancellationToken cancelToken, Action<double>? pfnProgress = null, eModelImplementation impl = eModelImplementation.GPU )
		{
			TaskCompletionSource<iModel> tcs = new TaskCompletionSource<iModel>();

			WaitCallback wcb = delegate ( object? state )
			{
				try
				{
					sLoadModelCallbacks callbacks = new sLoadModelCallbacks( cancelToken, pfnProgress );

					iModel model;
					NativeLogger.prologue();
					int hr = loadModel( path, impl, ref callbacks, out model );
					NativeLogger.throwForHR( hr );

					tcs.SetResult( model );
				}
				catch( Exception ex )
				{
					tcs.SetException( ex );
				}
			};

			ThreadPool.QueueUserWorkItem( wcb );
			return tcs.Task;
		}

		[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
		static extern int initMediaFoundation( [MarshalAs( UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof( Marshaler<iMediaFoundation> ) )] out iMediaFoundation mf );

		/// <summary>Initialize Media Foundation runtime</summary>
		public static iMediaFoundation initMediaFoundation()
		{
			iMediaFoundation mf;
			NativeLogger.prologue();
			int hr = initMediaFoundation( out mf );
			NativeLogger.throwForHR( hr );
			return mf;
		}

		// The .NET runtime uses UTF-16 for the strings, so we only need the Unicode version of this function.
		// The native DLL exports both Unicode and ASCII versions.
		[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
		static extern uint findLanguageKeyW( [MarshalAs( UnmanagedType.LPWStr )] string lang );

		/// <summary>Try to resolve language code string like <c>"en"</c>, <c>"pl"</c> or <c>"uk"</c> into the strongly-typed enum.</summary>
		/// <remarks>The function is case-sensitive, <c>"EN"</c> or <c>"UK"</c> gonna fail.</remarks>
		public static eLanguage? languageFromCode( string lang )
		{
			uint key = findLanguageKeyW( lang );
			if( key != uint.MaxValue )
				return (eLanguage)key;
			return null;
		}

		/// <summary>Set up delegate to receive log messages from the C++ library</summary>
		public static void setLogSink( eLogLevel lvl, eLoggerFlags flags = eLoggerFlags.SkipFormatMessage, pfnLogMessage? pfn = null )
		{
			NativeLogger.setup( lvl, flags, pfn );
		}
	}
}