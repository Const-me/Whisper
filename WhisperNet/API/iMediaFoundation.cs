using ComLight;
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Whisper.Internal;

namespace Whisper
{
	/// <summary>Exposes a small subset of MS Media Foundation framework.</summary>
	/// <remarks>That framework is a part of Windows OS, since Vista.</remarks>
	/// <seealso href="https://learn.microsoft.com/en-us/windows/win32/medfound/microsoft-media-foundation-sdk" />
	[ComInterface( "fb9763a5-d77d-4b6e-aff8-f494813cebd8", eMarshalDirection.ToManaged ), CustomConventions( typeof( NativeLogger ) )]
	public interface iMediaFoundation: IDisposable
	{
		/// <summary>Decode complete audio file into a new memory buffer.</summary>
		/// <returns>
		/// The method asks MF to resample and convert audio into the suitable type for the Whisper model.<br/>
		/// If the path is a video file, the method will decode the first audio track.
		/// </returns>
		[RetValIndex( 2 )]
		iAudioBuffer loadAudioFile( [MarshalAs( UnmanagedType.LPWStr )] string path, [MarshalAs( UnmanagedType.U1 )] bool stereo = false );

		/// <summary>Create a reader to stream the audio file from disk</summary>
		/// <returns>
		/// The method returns an object which can be used to decode the audio file incrementally.<br/>
		/// For long audio files, this saves both memory (no need for large uncompressed PCM buffer), and time (decode and transcribe run concurrently on different CPU threads).<br/>
		/// If the path is a video file, the implementation will use the first audio track.
		/// </returns>
		[RetValIndex( 2 )]
		iAudioReader openAudioFile( [MarshalAs( UnmanagedType.LPWStr )] string path, [MarshalAs( UnmanagedType.U1 )] bool stereo = false );

		/// <summary>Create a reader to decode audio file in memory</summary>
		/// <remarks>The method first copies the content into another internal buffer, then creates a streaming decoder</remarks>
		[RetValIndex( 3 ), EditorBrowsable( EditorBrowsableState.Never )]
		iAudioReader loadAudioFileData( IntPtr data, long size, [MarshalAs( UnmanagedType.U1 )] bool stereo );

		/// <summary>List capture devices</summary>
		void listCaptureDevices( [MarshalAs( UnmanagedType.FunctionPtr )] pfnFoundCaptureDevices pfn, IntPtr pv );

		/// <summary>Open audio capture device</summary>
		[RetValIndex( 2 )]
		iAudioCapture openCaptureDevice( [MarshalAs( UnmanagedType.LPWStr )] string endpoint, [In] ref sCaptureParams captureParams );
	}

	/// <summary>Extension methods for <see cref="iMediaFoundation" /> interface</summary>
	public static class MediaFoundationExt
	{
		/// <summary>Create a reader to decode audio file in memory</summary>
		/// <remarks>The method first copies the content into another internal buffer, then creates a streaming decoder</remarks>
		public static iAudioReader loadAudioFileData( this iMediaFoundation mf, ReadOnlySpan<byte> span, bool stereo = false )
		{
			unsafe
			{
				fixed( byte* rsi = span )
				{
					return mf.loadAudioFileData( (IntPtr)rsi, span.Length, stereo );
				}
			}
		}
	}
}