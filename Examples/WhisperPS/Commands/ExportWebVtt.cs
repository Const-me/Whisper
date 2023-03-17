using System;
using System.Globalization;
using System.IO;
using System.Management.Automation;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">Write transcribe results into WebVTT format.</para>
	/// <para type="description">The format is documented there: https://en.wikipedia.org/wiki/WebVTT</para>
	/// </summary>
	/// <example><code>Export-WebVTT $transcribeResults -path transcript.vtt</code></example>
	[Cmdlet( VerbsData.Export, "WebVTT" )]
	public sealed class ExportWebVTT: Cmdlet
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
		[Parameter( Mandatory = true )]
		public string path { get; set; }

		static string printTime( TimeSpan ts ) =>
			ts.ToString( "hh':'mm':'ss'.'fff", CultureInfo.InvariantCulture );

		/// <summary>Performs execution of the command</summary>
		protected override void ProcessRecord()
		{
			if( string.IsNullOrEmpty( path ) )
				throw new ArgumentException( "" );
			string dir = Path.GetDirectoryName( path );
			if( !string.IsNullOrEmpty( dir ) )
				Directory.CreateDirectory( dir );

			var segments = source.getResult().segments;
			using( var stream = File.CreateText( path ) )
			{
				stream.WriteLine( "WEBVTT" );
				stream.WriteLine();

				foreach( sSegment seg in segments )
				{
					string begin = printTime( seg.time.begin );
					string end = printTime( seg.time.end );
					stream.WriteLine( "{0} --> {1}", begin, end );
					stream.WriteLine( seg.text );
					stream.WriteLine();
				}
			}
		}
	}
}