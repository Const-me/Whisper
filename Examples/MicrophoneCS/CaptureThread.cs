using System.Runtime.ExceptionServices;
using Whisper;

namespace MicrophoneCS
{
	sealed class CaptureThread: CaptureCallbacks
	{
		public CaptureThread( CommandLineArgs args, Context context, iAudioCapture source )
		{
			callbacks = new TranscribeCallbacks( args );
			this.context = context;
			this.source = source;

			thread = new Thread( threadMain ) { Name = "Capture Thread" };
			Console.WriteLine( "Press any key to quit" );
			thread.Start();
		}

		static void readKeyCallback( object? state )
		{
			CaptureThread ct = ( state as CaptureThread ) ?? throw new ApplicationException();
			Console.ReadKey();
			ct.shouldQuit = true;
		}

		public void join()
		{
			ThreadPool.QueueUserWorkItem( readKeyCallback, this );
			thread.Join();
			edi?.Throw();
		}

		volatile bool shouldQuit = false;

		protected override bool shouldCancel( Context sender ) =>
			shouldQuit;

		protected override void captureStatusChanged( Context sender, eCaptureStatus status )
		{
			Console.WriteLine( $"CaptureStatusChanged: {status}" );
		}

		readonly TranscribeCallbacks callbacks;
		readonly Thread thread;
		readonly Context context;
		readonly iAudioCapture source;
		ExceptionDispatchInfo? edi = null;

		void threadMain()
		{
			try
			{
				context.runCapture( source, callbacks, this );
			}
			catch( Exception ex )
			{
				edi = ExceptionDispatchInfo.Capture( ex );
			}
		}
	}
}