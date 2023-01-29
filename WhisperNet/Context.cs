using System.Diagnostics;
using Whisper.Internal;
using Whisper.Internals;

namespace Whisper
{
	/// <summary>Stateful context, contains methods to transcribe audio</summary>
	public sealed class Context: IDisposable
	{
		iContext context;
		// Caching the results object here saves time spent in ComLight library creating these callable proxies over and over again for the same underlying C++ object
		readonly iTranscribeResult transcribeResult;
		sFullParams fullParams;
		sProgressSink progressSink;
		bool disposed = false;
		readonly Action<object> pfnBuffer, pfnStream;

		internal Context( Internal.iContext context )
		{
			this.context = context;
			transcribeResult = context.getResults( eResultFlags.None );
			fullParams = context.fullDefaultParams( eSamplingStrategy.Greedy );
			pfnBuffer = processBuffer;
			pfnStream = processStream;
			progressSink = default;
		}

		void IDisposable.Dispose()
		{
			if( disposed )
				return;
			disposed = true;
			context?.Dispose();
			GC.SuppressFinalize( this );
		}

		/// <summary>Adjustable parameters</summary>
		public ref Parameters parameters => ref fullParams.publicParams;

		void processBuffer( object buffer )
		{
			context.runFull( ref fullParams, (iAudioBuffer)buffer );
		}
		void processStream( object reader )
		{
			context.runStreamed( ref fullParams, ref progressSink, (iAudioReader)reader );
		}

		void runImpl( object source, Callbacks? callbacks, ReadOnlySpan<int> promptTokens, Action<object> pfn )
		{
			if( null != callbacks )
			{
				// TODO [very low, performance]: the following code creates 2 new GC-allocated objects on each call.
				// Possible to optimize by caching these function pointers in static readonly fields, and use another [ThreadStatic] field for the callbacks object
				fullParams.newSegmentCallback = delegate ( IntPtr ctx, int countNew, IntPtr userData )
				{
					return callbacks.newSegment( this, countNew );
				};

				fullParams.encoderBeginCallback = delegate ( IntPtr ctx, IntPtr userData )
				{
					return callbacks.encoderBegin( this );
				};
			}

			try
			{
				if( promptTokens.IsEmpty )
				{
					pfn( source );
					return;
				}
				unsafe
				{
					fixed( int* tokens = promptTokens )
					{
						fullParams.prompt_tokens = (IntPtr)tokens;
						fullParams.prompt_n_tokens = promptTokens.Length;
						pfn( source );
					}
				}
			}
			finally
			{
				// Reset these delegates.
				// Otherwise, this class will retain the callbacks object preventing it from being garbage collected.
				fullParams.newSegmentCallback = null;
				fullParams.encoderBeginCallback = null;

				fullParams.prompt_tokens = IntPtr.Zero;
				fullParams.prompt_n_tokens = 0;
			}
		}

		/// <summary>Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text</summary>
		public void runFull( iAudioBuffer buffer, Callbacks? callbacks, ReadOnlySpan<int> promptTokens )
		{
			runImpl( buffer, callbacks, promptTokens, pfnBuffer );
		}
		/// <summary>Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text</summary>
		public void runFull( iAudioBuffer buffer, Callbacks? callbacks = null ) =>
			runFull( buffer, callbacks, ReadOnlySpan<int>.Empty );
		/// <summary>Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text</summary>
		public void runFull( iAudioBuffer buffer, Callbacks? callbacks, int[]? promptTokens ) =>
			runFull( buffer, callbacks, promptTokens ?? ReadOnlySpan<int>.Empty );

		/// <summary>Run the entire model, streaming audio from the provided reader object</summary>
		public void runFull( iAudioReader reader, Callbacks? callbacks, Action<double>? pfnProgress, ReadOnlySpan<int> promptTokens )
		{
			if( null != pfnProgress )
			{
				progressSink.pfn = delegate ( double value, IntPtr context, IntPtr pv )
				{
					try
					{
						pfnProgress.Invoke( value );
						return 0;
					}
					catch( Exception ex )
					{
						return ex.HResult;
					}
				};
			}
			try
			{
				runImpl( reader, callbacks, promptTokens, pfnStream );
			}
			finally
			{
				progressSink.pfn = null;
			}
		}

		/// <summary>Run the entire model, streaming audio from the provided reader object</summary>
		public void runFull( iAudioReader reader, Action<double>? pfnProgress = null, Callbacks? callbacks = null ) =>
			runFull( reader, callbacks, pfnProgress, ReadOnlySpan<int>.Empty );

		/// <summary>Run the entire model, streaming audio from the provided reader object</summary>
		public void runFull( iAudioReader reader, Callbacks? callbacks, Action<double>? pfnProgress, int[]? promptTokens ) =>
			runFull( reader, callbacks, pfnProgress, promptTokens ?? ReadOnlySpan<int>.Empty );

		/// <summary>Get text results out of the context</summary>
		public TranscribeResult results( eResultFlags flags = eResultFlags.None )
		{
			if( flags.HasFlag( eResultFlags.NewObject ) )
				throw new ArgumentException();

			iTranscribeResult res = context.getResults( flags );
			Debug.Assert( ReferenceEquals( res, transcribeResult ) );
			return new TranscribeResult( res );
		}

		/// <summary>Print timing data</summary>
		public void timingsPrint() => context.timingsPrint();

		/// <summary>Reset timing data</summary>
		public void timingsReset() => context.timingsReset();

		/// <summary>Continuously process audio from microphone or a similar capture device</summary>
		/// <remarks>It’s recommended to call this method on a background thread.</remarks>
		public void runCapture( iAudioCapture capture, Callbacks? callbacks, CaptureCallbacks? captureCallbacks )
		{
			if( null != callbacks )
			{
				// TODO [very low, performance]: the following code creates 2 new GC-allocated objects on each call.
				// Possible to optimize by caching these function pointers in static readonly fields, and use another [ThreadStatic] field for the callbacks object
				fullParams.newSegmentCallback = delegate ( IntPtr ctx, int countNew, IntPtr userData )
				{
					return callbacks.newSegment( this, countNew );
				};

				fullParams.encoderBeginCallback = delegate ( IntPtr ctx, IntPtr userData )
				{
					return callbacks.encoderBegin( this );
				};
			}

			try
			{
				sCaptureCallbacks cc = default;
				if( captureCallbacks != null )
				{
					cc.shouldCancel = captureCallbacks.cancel( this );
					cc.captureStatus = captureCallbacks.status( this );
				}
				context.runCapture( ref fullParams, ref cc, capture );
			}
			finally
			{
				// Reset these delegates.
				// Otherwise, this class will retain the callbacks object preventing it from being garbage collected.
				fullParams.newSegmentCallback = null;
				fullParams.encoderBeginCallback = null;

				fullParams.prompt_tokens = IntPtr.Zero;
				fullParams.prompt_n_tokens = 0;
			}
		}

		/// <summary>Try to detect speaker by comparing channels of the stereo PCM data</summary>
		/// <remarks>
		/// <para>The feature requires stereo PCM data.<br/>Pass <c>stereo=true</c> to <see cref="iMediaFoundation.loadAudioFile" /> or <see cref="iMediaFoundation.openAudioFile"/> methods,<br/>
		/// or <see cref="eCaptureFlags.Stereo" /> to <see cref="iMediaFoundation.openCaptureDevice" /> method.</para>
		/// <para>It seems to work fine with <a href="https://www.bluemic.com/en-us/products/yeti/">Blue Yeti</a> microphone,
		/// after switched the microphone to Stereo pattern.<br/> With recorded sounds however, the performance varies depending on the recording.</para>
		/// </remarks>
		public eSpeakerChannel detectSpeaker( sTimeInterval interval ) =>
			context.detectSpeaker( ref interval );
	}
}