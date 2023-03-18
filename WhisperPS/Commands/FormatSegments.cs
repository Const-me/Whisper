using System.Management.Automation;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">Format transcribe results as a sequence of segments.</para>
	/// <para type="description">Each segment has a pair of timestamps, and the text</para>
	/// </summary>
	/// <example><code>Format-Segments $transcribeResults</code></example>
	[Cmdlet( VerbsCommon.Format, "Segments" )]
	public sealed class FormatSegments: Cmdlet
	{
		/// <summary>
		/// <para type="synopsis">Transcribe result produced by <see cref="TranscribeFile" /></para>
		/// <para type="inputType">It requires the value of the correct type</para>
		/// </summary>
		[Parameter( Mandatory = true, ValueFromPipeline = true )]
		public Transcription source { get; set; }

		/// <summary>Performs execution of the command</summary>
		protected override void ProcessRecord()
		{
			var res = source.getResult();
			foreach( var seg in res.segments )
			{
				Segment obj = new Segment( seg );
				WriteObject( obj );
			}
		}
	}
}