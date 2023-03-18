using System.IO;
using System.Management.Automation;
using Whisper.Internal;
using Whisper.Internals;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">Transcribe audio file from disk</para>
	/// </summary>
	/// <example>
	/// <code>$res = Transcribe-File -model $m -path C:\Temp\SampleClips\jfk.wav</code>
	/// </example>
	[Cmdlet( "Transcribe", "File" )]
	public sealed class TranscribeFile: TranscribeBase
	{
		/// <summary>
		/// <para type="synopsis">Path to the input file</para>
		/// <para type="description">The command supports most audio and video formats, with the notable exception of Ogg Vorbis.</para>
		/// </summary>
		[Parameter( Mandatory = true, Position = 1 ), ValidateNotNullOrEmpty]
		public string path { get; set; }

		protected override void BeginProcessing()
		{
			if( null == model )
				throw new PSArgumentNullException( nameof( model ) );
			validateLanguage();
		}

		/// <summary>Performs execution of the command</summary>
		protected override void ProcessRecord()
		{
			string path = this.absolutePath( this.path );
			if( !File.Exists( path ) )
				throw new FileNotFoundException( "Input file not found" );

			using( var log = this.setupLog() )
			using( iContext context = model.model.createContextInternal() )
			using( iAudioReader reader = model.mf.openAudioFile( path ) )
			{
				sFullParams fullParams = context.fullDefaultParams( eSamplingStrategy.Greedy );
				applyParams( ref fullParams.publicParams );
				sProgressSink progressSink = makeProgressSink();
				context.runStreamed( ref fullParams, ref progressSink, reader );
				var obj = context.getResults( eResultFlags.Tokens | eResultFlags.NewObject );
				WriteObject( new Transcription( obj ) );
			}
		}
	}
}