using ComLight;
using System.Runtime.InteropServices;

namespace Whisper
{
	/// <summary>A buffer with a chunk of audio.</summary>
	/// <remarks>Note the interface supports both marshaling directions.<br/>
	/// I have not tested, but you should be able to implement this interface in C#, to supply PCM audio data to the native code</remarks>
	[ComInterface( "013583aa-c9eb-42bc-83db-633c2c317051", eMarshalDirection.BothWays )]
	public interface iAudioBuffer: IDisposable
	{
		/// <summary>Count of samples in the buffer</summary>
		int countSamples();

		/// <summary>Unmanaged pointer to the internal buffer containing single-channel FP32 samples.</summary>
		/// <remarks>If you implementing this interface in C# and your audio data is on the managed heap, use <see cref="GCHandle" /> to make sure it doesn't move.<br/>
		/// Or better yet, move the data to unmanaged buffer allocated with <see cref="Marshal.AllocHGlobal(int)" /> or <see cref="Marshal.AllocCoTaskMem(int)" /> method.</remarks>
		IntPtr getPcmMono();

		/// <summary>Unmanaged pointer to the internal buffer containing stereo FP32 samples.</summary>
		/// <remarks>When the buffer doesn’t have stereo data, the method gonna return <see cref="IntPtr.Zero" />.</remarks>
		IntPtr getPcmStereo();

		/// <summary>Start time of the buffer, relative to the start of the media</summary>
		void getTime( out TimeSpan time );
	}
}