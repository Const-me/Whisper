using System;
using System.IO;
using System.Management.Automation;

namespace Whisper
{
	/// <summary>Base class for commands which export results into some text-based format</summary>
	public abstract class ExportBase: PSCmdlet
	{
		/// <summary>
		/// <para type="synopsis">Transcribe result produced by <see cref="TranscribeFile" /></para>
		/// <para type="inputType">It requires the value of the correct type</para>
		/// </summary>
		[Parameter( Mandatory = true, ValueFromPipeline = true )]
		public Transcription source { get; set; }

		/// <summary>
		/// <para type="synopsis">Output file to write</para>
		/// </summary>
		[Parameter( Mandatory = true, Position = 0 ), ValidateNotNullOrEmpty]
		public string path { get; set; }

		/// <summary>Performs execution of the command</summary>
		protected override void ProcessRecord()
		{
			string path = this.absolutePath( this.path );
			string dir = Path.GetDirectoryName( path );
			Directory.CreateDirectory( dir );
			if( File.Exists( path ) )
				if( !ShouldContinue( $"Overwrite \"{path}\" ?", "The output file already exists" ) )
					return;

			var results = source.getResult();
			using( var stream = File.CreateText( path ) )
				write( stream, results );
		}

		/// <summary>Actual implementation</summary>
		protected abstract void write( StreamWriter stream, TranscribeResult transcribeResult );
	}
}