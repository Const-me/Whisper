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
		/// <para type="synopsis">Whisper model in VRAM</para>
		/// <para type="description">Use <c>Import-WhisperModel</c> command to load the model from disk</para>
		/// </summary>
		[Parameter( Mandatory = true )]
		public Model model { get; set; }

		/// <summary>
		/// <para type="synopsis">Path to the input file</para>
		/// <para type="description">The command supports most audio and video formats, with the notable exception of Ogg Vorbis.</para>
		/// </summary>
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

		/// <summary>Performs execution of the command</summary>
		protected override void ProcessRecord()
		{
			using( var log = this.setupLog() )
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