using ComLight;
using System;
using System.Runtime.InteropServices;
using Whisper.Internals;

namespace Whisper.Internal
{
    /// <summary>Stateful context, contains methods to transcribe audio</summary>
    [ComInterface( "b9956374-3b18-4943-90f2-2ab18a404537", eMarshalDirection.ToManaged ), CustomConventions( typeof( NativeLogger ) )]
	public interface iContext: IDisposable
	{
		/// <summary>Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text</summary>
		void runFull( [In] ref sFullParams @params, iAudioBuffer buffer );

		/// <summary>Run the entire model, streaming audio from the provided reader object</summary>
		void runStreamed( [In] ref sFullParams @params, [In] ref sProgressSink progressSink, iAudioReader reader );

		/// <summary>Continuously process audio from microphone or a similar capture device</summary>
		void runCapture( [In] ref sFullParams @params, [In] ref sCaptureCallbacks callbacks, iAudioCapture reader );

		/// <summary>Get text results out of the context</summary>
		[RetValIndex( 1 )]
		iTranscribeResult getResults( eResultFlags flags );

		/// <summary>Try to detect speaker by comparing channels of the stereo PCM data</summary>
		[RetValIndex( 1 )]
		eSpeakerChannel detectSpeaker( [In] ref sTimeInterval interval );

		/// <summary>Get the model which was used to create this context</summary>
		[RetValIndex]
		iModel getModel();

		/// <summary>Full the default parameters of the model, for the specified sampling strategy</summary>
		[RetValIndex( 1 )]
		sFullParams fullDefaultParams( eSamplingStrategy strategy );

		/// <summary>Print timing data</summary>
		void timingsPrint();
		/// <summary>Reset timing data</summary>
		void timingsReset();
	}
}