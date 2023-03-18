using ComLight;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Whisper.Internal;

namespace Whisper
{
	/// <summary>Factory methods implemented by the C++ DLL</summary>
	static class Library
	{
		static Library()
		{
			if( Environment.OSVersion.Platform != PlatformID.Win32NT )
				throw new ApplicationException( "This library requires Windows OS" );
			if( !Environment.Is64BitProcess )
				throw new ApplicationException( "This library only works in 64-bit processes" );
			if( RuntimeInformation.ProcessArchitecture != Architecture.X64 )
				throw new ApplicationException( "This library requires a processor with AMD64 instruction set" );
			NativeLogger.startup();
		}

		const string dll = "Whisper.dll";

		[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = false )]
		internal static extern void setupLogger( [In] ref sLoggerSetup setup );

		[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
		static extern int loadModel( [MarshalAs( UnmanagedType.LPWStr )] string path,
			[In] ref sModelSetup setup, [In] ref sLoadModelCallbacks callbacks,
			[MarshalAs( UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof( Marshaler<iModel> ) )] out iModel model );

		/// <summary>Load Whisper model from GGML file on disk</summary>
		/// <remarks>Models are large, depending on user’s disk speed this might take a while, and this function blocks the calling thread.<br/>
		/// Consider <see cref="loadModelAsync" /> instead.</remarks>
		/// <seealso href="https://huggingface.co/datasets/ggerganov/whisper.cpp" />
		public static iModel loadModel( string path, eGpuModelFlags flags = eGpuModelFlags.None,
			string adapter = null, Action<double> progress=null, eModelImplementation impl = eModelImplementation.GPU )
		{
			iModel model;
			sModelSetup setup = new sModelSetup( flags, impl, adapter );
			sLoadModelCallbacks callbacks = new sLoadModelCallbacks( CancellationToken.None, progress );
			NativeLogger.prologue();
			int hr = loadModel( path, ref setup, ref callbacks, out model );
			NativeLogger.throwForHR( hr );
			return model;
		}

		/// <summary>Load Whisper model on a background thread, with optional progress reporting and cancellation</summary>
		public static Task<iModel> loadModelAsync( string path, CancellationToken cancelToken,
			eGpuModelFlags flags = eGpuModelFlags.None, string adapter = null,
			Action<double> pfnProgress = null, eModelImplementation impl = eModelImplementation.GPU )
		{
			TaskCompletionSource<iModel> tcs = new TaskCompletionSource<iModel>();

			WaitCallback wcb = delegate ( object state )
			{
				try
				{
					sModelSetup setup = new sModelSetup( flags, impl, adapter );
					sLoadModelCallbacks callbacks = new sLoadModelCallbacks( cancelToken, pfnProgress );

					iModel model;
					NativeLogger.prologue();
					int hr = loadModel( path, ref setup, ref callbacks, out model );
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
		public static void setLogSink( eLogLevel lvl, eLoggerFlags flags = eLoggerFlags.SkipFormatMessage, pfnLogMessage pfn = null )
		{
			NativeLogger.setup( lvl, flags, pfn );
		}

		[DllImport( dll, CallingConvention = RuntimeClass.defaultCallingConvention, PreserveSig = true )]
		static extern int listGPUs( [MarshalAs( UnmanagedType.FunctionPtr )] pfnListAdapters pfn, IntPtr pv );

		/// <summary>Enumerate graphics adapters on this computer, and return their names.</summary>
		/// <remarks>To manually select the GPU to use for the inference, pass one of these names to<br/>
		/// <see cref="loadModel(string, eGpuModelFlags, string, eModelImplementation)" /> or <br/>
		/// <see cref="loadModelAsync(string, CancellationToken, eGpuModelFlags, string, Action{double}, eModelImplementation)" /> factory function.</remarks>
		public static string[] listGraphicAdapters()
		{
			List<string> list = new List<string>();
			pfnListAdapters pfn = delegate ( string name, IntPtr pv )
			{
				Debug.Assert( pv == IntPtr.Zero );
				list.Add( name );
			};

			NativeLogger.prologue();
			int hr = listGPUs( pfn, IntPtr.Zero );
			NativeLogger.throwForHR( hr );

			return list.ToArray();
		}
	}
}