using ComLight;
using System;

namespace Whisper
{
	/// <summary>Audio stream reader object</summary>
	/// <remarks>The implementation is forward-only, and these objects aren’t reusable.<br/>
	/// To read an audio file multiple time, dispose this object, and create a new one from the same source file.</remarks>
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