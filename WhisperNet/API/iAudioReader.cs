using ComLight;

namespace Whisper
{
	/// <summary>Audio stream reader object</summary>
	/// <remarks>The implementation is forward-only, and these objects ain’t reusable.<br/>
	/// To read a source file multiple time, dispose and re-create the reader.</remarks>
	[ComInterface( "35b988da-04a6-476a-a193-d8891d5dc390", eMarshalDirection.ToManaged )]
	public interface iAudioReader: IDisposable
	{
		/// <summary>Get duration of the media file</summary>
		[RetValIndex]
		TimeSpan getDuration();
	}

	/// <summary>Audio capture reader object</summary>
	/// <remarks>This interface has no public methods callable from C#.<br/>
	/// It’s only here to pass data between different functions implemented in C++.</remarks>
	[ComInterface( "747752c2-d9fd-40df-8847-583c781bf013", eMarshalDirection.ToManaged )]
	public interface iAudioCapture: IDisposable
	{
	}
}