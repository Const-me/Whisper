using System.Globalization;

namespace PerfSummary
{
	static class Summary
	{
		public static void print( LogData[] logs, string folder )
		{
			string path = Path.Combine( folder, "summary.tsv" );
			using var writer = File.CreateText( path );

			string[] header = new string[]
			{
				"Audio Clip", "Model", "GPU",
				"Total, sec", "Relative speed",
				"Encode, sec", "Decode, sec",
				"RAM, MB", "VRAM, MB"
			};
			writer.fields( header );

			foreach( var item in logs )
				writer.fields( item.makeFields() );
		}

		static IEnumerable<string> makeFields( this LogData log )
		{
			yield return log.clip();
			yield return log.name.model.ToString();
			yield return log.gpu();
			yield return log.runComplete.print();
			yield return log.relative();
			yield return log.encode.print();
			yield return log.decode.print();
			yield return log.ram.print();
			yield return log.vram.print();
		}

		static string print( this double v ) =>
			v.ToString( CultureInfo.InvariantCulture );

		static string relative( this LogData log )
		{
			double duration;
			switch( log.name.clip )
			{
				case eInputClip.jfk:
					duration = 11;
					break;
				case eInputClip.columbia:
					duration = TimeSpan.FromTicks( 1987620000 ).TotalSeconds;
					break;
				default:
					throw new NotImplementedException();
			}

			double rel = duration / log.runComplete;
			return rel.ToString( "F4", CultureInfo.InvariantCulture );
		}

		static string clip( this LogData log )
		{
			return log.name.clip switch
			{
				eInputClip.jfk => "jfk.wav",
				eInputClip.columbia => "columbia.wma",
				 _ => throw new ArgumentException()
			};
		}

		static string gpu( this LogData log )
		{
			return log.name.gpu switch
			{
				"1080ti" => "GeForce 1080Ti",
				"1650" => "GeForce 1650",
				"vega7" => "Ryzen 5 5600U",
				"vega8" => "Ryzen 7 5700G",
				_ => log.name.gpu
			};
		}

		static void fields( this StreamWriter writer, IEnumerable<string> fields )
		{
			string line = string.Join( "\t", fields );
			writer.WriteLine( line );
		}
	}
}