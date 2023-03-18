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
	/// <example><code>Export-Text $transcribeResults -path transcript.txt -timestamps</code></example>
	[Cmdlet( VerbsData.Export, "Text" )]
	public sealed class ExportText: ExportBase
	{
		/// <summary>
		/// <para type="synopsis">Specify this switch to include timestamps</para>
		/// </summary>
		[Parameter]
		public SwitchParameter timestamps { get; set; }

		static string printTime( TimeSpan ts ) =>
			ts.ToString( "hh':'mm':'ss'.'fff", CultureInfo.InvariantCulture );

		/// <summary>Write that text</summary>
		protected override void write( StreamWriter stream, TranscribeResult transcribeResult )
		{
			foreach( sSegment seg in transcribeResult.segments )
			{
				if( timestamps )
				{
					string begin = printTime( seg.time.begin );
					string end = printTime( seg.time.end );
					stream.Write( "[{0} --> {1}]  ", begin, end );
				}
				stream.WriteLine( seg.text.Trim() );
			}
		}
	}
}