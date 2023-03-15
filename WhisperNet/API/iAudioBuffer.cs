using ComLight;
using System;
using System.Runtime.InteropServices;

namespace Whisper
{
	/// <summary>A buffer with a chunk of audio.</summary>
	/// <remarks>Note the interface supports both marshaling directions.<br/>
	/// You can implement this interface in C#, to supply PCM audio data to the native code.</remarks>
	/// <seealso href="https://gist.github.com/Const-me/082c8d96eb10b76058c5dd9c68b5bfe1" />
	[ComInterface( "013583aa-c9eb-42bc-83db-633c2c317051", eMarshalDirection.BothWays )]
	public interface iAudioBuffer: IDisposable
	{
		/// <summary>Count of samples in the buffer, equal to ( length in seconds ) * 16000</summary>
		int countSamples();

		/// <summary>Unmanaged pointer to the internal buffer with single-channel <c>float</c> PCM samples @ 16 kHz sample rate.</summary>
		/// <remarks>If you implementing this interface in C# and your audio data is on the managed heap, use <see cref="GCHandle" /> to make sure it doesn't move.<br/>
		/// Or better yet, move the data to unmanaged buffer allocated with <see cref="Marshal.AllocHGlobal(int)" /> or <see cref="Marshal.AllocCoTaskMem(int)" /> method.</remarks>
		IntPtr getPcmMono();

		/// <summary>Unmanaged pointer to the internal buffer with interleaved stereo <c>float</c> PCM samples @ 16 kHz sample rate.</summary>
		/// <remarks>When the buffer doesn’t have stereo data, the method should return <see cref="IntPtr.Zero" />.</remarks>
		IntPtr getPcmStereo();

		/// <summary>Start time of the buffer, relative to the start of the media.</summary>
		/// <remarks>The value is used to produce timestamps in <see cref="sSegment.time" /> and <see cref="sToken.time" /> fields.</remarks>
		void getTime( out TimeSpan time );
	}
}