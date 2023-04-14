using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Whisper.Internal;

namespace Whisper
{
	/// <summary>Extension methods of these COM interfaces</summary>
	public static class ExtensionMethods
	{
		/// <summary>Create a context to transcribe audio with this model</summary>
		public static Context createContext( this iModel model )
		{
			iContext ctx = model.createContextInternal();
			return new Context( ctx );
		}

		/// <summary>Convert language into a short ID string, like <c>"en"</c></summary>
		[SkipLocalsInit]
		public static string getCode( this eLanguage lang )
		{
			unsafe
			{
				sbyte* ptr = stackalloc sbyte[ 5 ];
				*(uint*)ptr = (uint)lang;
				ptr[ 4 ] = 0;
				return new string( ptr );
			}
		}

		/// <summary>Resolve integer token ID into string.</summary>
		/// <remarks>If the token ID was not found in the model, the method returns null without raising exceptions.</remarks>
		public static string? stringFromToken( this iModel model, int idToken ) =>
			Marshal.PtrToStringUTF8( model.stringFromTokenInternal( idToken ) );

		/// <summary>List capture devices</summary>
		public static CaptureDeviceId[]? listCaptureDevices( this iMediaFoundation mf )
		{
			CaptureDeviceId[]? result = null;

			pfnFoundCaptureDevices pfn = delegate ( int len, sCaptureDevice[]? arr, IntPtr pv )
			{
				try
				{
					if( len == 0 || arr == null )
						return 1;
					Debug.Assert( len == arr.Length );

					result = new CaptureDeviceId[ len ];
					for( int i = 0; i < len; i++ )
						result[ i ] = new CaptureDeviceId( arr[ i ] );
					return 0;
				}
				catch( Exception ex )
				{
					NativeLogger.captureException( ex );
					return ex.HResult;
				}
			};

			mf.listCaptureDevices( pfn, IntPtr.Zero );

			return result;
		}

		/// <summary>Open audio capture device</summary>
		public static iAudioCapture openCaptureDevice( this iMediaFoundation mf, in CaptureDeviceId id, in sCaptureParams? cp = null )
		{
			sCaptureParams captureParams = cp ?? new sCaptureParams();
			return mf.openCaptureDevice( id.endpoint, ref captureParams );
		}

		/// <summary>Convert the provided text into tokens</summary>
		public static int[]? tokenize( this iModel model, string text )
		{
			int[]? result = null;
			pfnDecodedTokens pfn = delegate ( int[] arr, int length, IntPtr pv )
			{
				result = arr;
			};
			model.tokenize( text, pfn, IntPtr.Zero );
			return result;
		}
	}
}