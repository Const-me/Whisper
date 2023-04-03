using System;
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
		[Parameter( Mandatory = true, Position = 1, ValueFromPipeline = true ), ValidateNotNullOrEmpty]
		public string path { get; set; }

		int[] promptTokens = null;

		protected override void BeginProcessing()
		{
			if( null == model )
				throw new PSArgumentNullException( nameof( model ) );

			validateLanguage();

			promptTokens = null;
			if( !string.IsNullOrEmpty( prompt ) )
				promptTokens = tokenize( model.model, prompt );
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
				sProgressSink progressSink = makeProgressSink( $"Transcribing {path}" );

				if( null == promptTokens || promptTokens.Length <= 0 )
					context.runStreamed( ref fullParams, ref progressSink, reader );
				else
				{
					unsafe
					{
						fixed( int* p = promptTokens )
						{
							fullParams.prompt_tokens = (IntPtr)p;
							fullParams.prompt_n_tokens = promptTokens.Length;
							context.runStreamed( ref fullParams, ref progressSink, reader );
						}
					}
				}

				var obj = context.getResults( eResultFlags.Tokens | eResultFlags.NewObject );
				WriteObject( new Transcription( path, obj ) );
			}
		}
	}
}