using System;
using System.Globalization;
using System.IO;
using System.Management.Automation;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">Write transcribe results into SubRip format.</para>
	/// <para type="description">The format is documented there: https://en.wikipedia.org/wiki/SubRip#SubRip_file_format</para>
	/// </summary>
	/// <example><code>Export-SubRip $transcribeResults -path transcript.srt</code></example>
	[Cmdlet( VerbsData.Export, "SubRip" )]
	public sealed class ExportSubrip: ExportBase
	{
		/// <summary>
		/// <para type="synopsis">Optional integer offset to the indices</para>
		/// </summary>
		[Parameter]
		public int offset { get; set; } = 0;

		static string printTimeWithComma( TimeSpan ts ) =>
			ts.ToString( "hh':'mm':'ss','fff", CultureInfo.InvariantCulture );

		/// <summary>Write that text</summary>
		protected override void write( StreamWriter stream, TranscribeResult transcribeResult )
		{
			var segments = transcribeResult.segments;

			for( int i = 0; i < segments.Length; i++ )
			{
				stream.WriteLine( i + 1 + offset );
				sSegment seg = segments[ i ];
				string begin = printTimeWithComma( seg.time.begin );
				string end = printTimeWithComma( seg.time.end );
				stream.WriteLine( "{0} --> {1}", begin, end );
				stream.WriteLine( seg.text.Trim() );
				stream.WriteLine();
			}
		}
	}
}