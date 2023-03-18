using System;
using System.Management.Automation;

namespace Whisper
{
	static class MiscUtils
	{
		public static void writeProgress( this Cmdlet cmd, double progressValue, string what )
		{
			int percents = (int)Math.Round( progressValue * 100.0 );
			percents = Math.Max( 0, percents );
			percents = Math.Min( percents, 100 );

			ProgressRecord rec = new ProgressRecord( 0, what, "please wait…" );
			rec.PercentComplete = percents;
			cmd.WriteProgress( rec );
		}
	}
}