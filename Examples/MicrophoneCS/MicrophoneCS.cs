using Whisper;

namespace MicrophoneCS
{
	static class Program
	{
		static int Main( string[] args )
		{
			try
			{
				CommandLineArgs cla;
				try
				{
					cla = new CommandLineArgs( args );
				}
				catch( OperationCanceledException )
				{
					return 1;
				}
				const eLoggerFlags loggerFlags = eLoggerFlags.UseStandardError | eLoggerFlags.SkipFormatMessage;
				Library.setLogSink( eLogLevel.Debug, loggerFlags );

				using iMediaFoundation mf = Library.initMediaFoundation();
				CaptureDeviceId[] devices = mf.listCaptureDevices() ??
					throw new ApplicationException( "This computer has no audio capture devices" );

				if( cla.listDevices )
				{
					for( int i = 0; i < devices.Length; i++ )
						Console.WriteLine( "#{0}: {1}", i, devices[ i ].displayName );
					return 0;
				}
				if( cla.captureDeviceIndex < 0 || cla.captureDeviceIndex >= devices.Length )
					throw new ApplicationException( $"Capture device index is out of range; the valid range is [ 0 .. {devices.Length - 1} ]" );

				sCaptureParams cp = new sCaptureParams( true );
				if( cla.diarize )
					cp.flags |= eCaptureFlags.Stereo;
				using iAudioCapture captureDev = mf.openCaptureDevice( devices[ cla.captureDeviceIndex ], cp );

				using iModel model = Library.loadModel( cla.model );
				using Context context = model.createContext();
				cla.apply( ref context.parameters );

				CaptureThread thread = new CaptureThread( cla, context, captureDev );
				thread.join();

				context.timingsPrint();
				return 0;
			}
			catch( Exception ex )
			{
				// Console.WriteLine( ex.Message );
				Console.WriteLine( ex.ToString() );
				return ex.HResult;
			}
		}
	}
}