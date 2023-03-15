using System.IO;
using System.Management.Automation;
using Whisper.Internal;
using Whisper.Internals;

namespace Whisper
{
	[Cmdlet( "Transcribe", "File" )]
	public sealed class TranscribeFile: TranscribeBase
	{
		[Parameter( Mandatory = true )]
		public Model model { get; set; }

		[Parameter( Mandatory = true )]
		public string path { get; set; }

		protected override void BeginProcessing()
		{
			if( null == model )
				throw new PSArgumentNullException( nameof( model ) );
			if( string.IsNullOrEmpty( path ) )
				throw new PSArgumentNullException( nameof( path ) );
			if( !File.Exists( path ) )
				throw new FileNotFoundException( "Audio file not found", path );

			validateLanguage();
		}

		protected override void ProcessRecord()
		{
			using( iContext context = model.model.createContextInternal() ) 
			using( iAudioReader reader = model.mf.openAudioFile( path ) )
			{
				sFullParams fullParams = context.fullDefaultParams( eSamplingStrategy.Greedy );
				sProgressSink progressSink = default;
				applyParams( ref fullParams.publicParams );
				context.runStreamed( ref fullParams, ref progressSink, reader );
				var obj = context.getResults( eResultFlags.Tokens | eResultFlags.NewObject );
				WriteObject( new Transcription( obj ) );
			}
		}
	}
}