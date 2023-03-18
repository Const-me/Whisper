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
	public sealed class ExportWebVTT: ExportBase
	{
		static string printTime( TimeSpan ts ) =>
			ts.ToString( "hh':'mm':'ss'.'fff", CultureInfo.InvariantCulture );

		/// <summary>Write that text</summary>
		protected override void write( StreamWriter stream, TranscribeResult transcribeResult )
		{
			var segments = transcribeResult.segments;

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