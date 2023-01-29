using Whisper;

namespace TranscribeCS
{
	static class Program
	{
		static readonly bool streamAudio = true;

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

				using iModel model = Library.loadModel( cla.model );
				using Context context = model.createContext();
				cla.apply( ref context.parameters );
				using iMediaFoundation mf = Library.initMediaFoundation();
				Transcribe transcribe = new Transcribe( cla );

				foreach( string audioFile in cla.fileNames )
				{
					if( streamAudio )
					{
						using iAudioReader reader = mf.openAudioFile( audioFile, cla.diarize );
						context.runFull( reader, transcribe, null, cla.prompt );
					}
					else
					{
						using iAudioBuffer buffer = mf.loadAudioFile( audioFile, cla.diarize );
						context.runFull( buffer, transcribe, cla.prompt );
					}
					// When asked to, produce these text files
					if( cla.output_txt )
						writeTextFile( context, audioFile );
					if( cla.output_srt )
						writeSubRip( context, audioFile, cla );
					if( cla.output_vtt )
						writeWebVTT( context, audioFile );
				}

				context.timingsPrint();
				return 0;
			}
			catch( Exception ex )
			{
				Console.WriteLine( ex.Message );
				return ex.HResult;
			}
		}

		static void writeTextFile( Context context, string audioPath )
		{
			using var stream = File.CreateText( Path.ChangeExtension( audioPath, ".txt" ) );
			foreach( sSegment seg in context.results().segments )
				stream.WriteLine( seg.text );
		}

		static void writeSubRip( Context context, string audioPath, CommandLineArgs cliArgs )
		{
			using var stream = File.CreateText( Path.ChangeExtension( audioPath, ".srt" ) );
			var segments = context.results( eResultFlags.Timestamps ).segments;

			for( int i = 0; i < segments.Length; i++ )
			{
				stream.WriteLine( i + 1 + cliArgs.offset_n );
				sSegment seg = segments[ i ];
				string begin = Transcribe.printTimeWithComma( seg.time.begin );
				string end = Transcribe.printTimeWithComma( seg.time.end );
				stream.WriteLine( "{0} --> {1}", begin, end );
				stream.WriteLine( seg.text );
				stream.WriteLine();
			}
		}

		static void writeWebVTT( Context context, string audioPath )
		{
			using var stream = File.CreateText( Path.ChangeExtension( audioPath, ".vtt" ) );
			stream.WriteLine( "WEBVTT" );
			stream.WriteLine();

			foreach( sSegment seg in context.results( eResultFlags.Timestamps ).segments )
			{
				string begin = Transcribe.printTime( seg.time.begin );
				string end = Transcribe.printTime( seg.time.end );
				stream.WriteLine( "{0} --> {1}", begin, end );
				stream.WriteLine( seg.text );
				stream.WriteLine();
			}
		}
	}
}