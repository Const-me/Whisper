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
	public sealed class ExportSubrip: Cmdlet
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

		/// <summary>
		/// <para type="synopsis">Optional integer offset to the indices</para>
		/// </summary>
		[Parameter]
		public int offset { get; set; } = 0;

		static string printTimeWithComma( TimeSpan ts ) =>
			ts.ToString( "hh':'mm':'ss','fff", CultureInfo.InvariantCulture );

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
}