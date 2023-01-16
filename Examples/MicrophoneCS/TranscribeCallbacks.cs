using System.Globalization;
using Whisper;

namespace MicrophoneCS
{
	/// <summary>Implementation of Callbacks abstract class, to print these segments as soon as they’re produced by the library.</summary>
	sealed class TranscribeCallbacks: Callbacks
	{
		readonly CommandLineArgs args;
		readonly eResultFlags resultFlags;

		public TranscribeCallbacks( CommandLineArgs args )
		{
			this.args = args;
			resultFlags = args.resultFlags();
			Console.OutputEncoding = System.Text.Encoding.UTF8;
		}

		// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
		// Lowest is red, middle is yellow, highest is green.
		readonly string[] k_colors = new string[]
		{
			"\x1B[38;5;196m", "\x1B[38;5;202m", "\x1B[38;5;208m", "\x1B[38;5;214m", "\x1B[38;5;220m",
			"\x1B[38;5;226m", "\x1B[38;5;190m", "\x1B[38;5;154m", "\x1B[38;5;118m", "\x1B[38;5;82m"
		};

		int colorIndex( in sToken tok )
		{
			float p = tok.probability;
			float p3 = p * p * p;
			int col = (int)( p3 * k_colors.Length );
			col = Math.Clamp( col, 0, k_colors.Length - 1 );
			return col;
		}

		public static string printTime( TimeSpan ts ) =>
			ts.ToString( "hh':'mm':'ss'.'fff", CultureInfo.InvariantCulture );
		public static string printTimeWithComma( TimeSpan ts ) =>
			ts.ToString( "hh':'mm':'ss','fff", CultureInfo.InvariantCulture );

		protected override void onNewSegment( Context sender, int countNew )
		{
			TranscribeResult res = sender.results( resultFlags );
			ReadOnlySpan<sToken> tokens = res.tokens;

			int s0 = res.segments.Length - countNew;
			if( s0 == 0 )
				Console.WriteLine();

			for( int i = s0; i < res.segments.Length; i++ )
			{
				sSegment seg = res.segments[ i ];

				if( args.no_timestamps )
				{
					if( args.print_colors && AnsiCodes.enabled )
					{
						foreach( sToken tok in res.getTokens( seg ) )
						{
							if( !args.print_special && tok.hasFlag( eTokenFlags.Special ) )
								continue;
							Console.Write( "{0}{1}{2}", k_colors[ colorIndex( tok ) ], tok.text, "\x1B[0m" );
						}
					}
					else
						Console.Write( seg.text );
					Console.Out.Flush();
					continue;
				}

				string speaker = "";
#if false
				if( args.diarize && pcmf32s.size() == 2 )
				{
					const size_t n_samples = pcmf32s[ 0 ].size();
					const int64_t is0 = SourceAudio::sampleFromTimestamp( seg.time.begin, n_samples );
					const int64_t is1 = SourceAudio::sampleFromTimestamp( seg.time.end, n_samples );

					double energy0 = 0.0f;
					double energy1 = 0.0f;

					for( int64_t j = is0; j < is1; j++ )
					{
						energy0 += fabs( pcmf32s[ 0 ][ j ] );
						energy1 += fabs( pcmf32s[ 1 ][ j ] );
					}

					if( energy0 > 1.1 * energy1 )
						speaker = "(speaker 0)";
					else if( energy1 > 1.1 * energy0 )
						speaker = "(speaker 1)";
					else
						speaker = "(speaker ?)";

					//printf("is0 = %lld, is1 = %lld, energy0 = %f, energy1 = %f, %s\n", is0, is1, energy0, energy1, speaker.c_str());
				}
#endif
				if( args.print_colors && AnsiCodes.enabled  )
				{
					Console.Write( "[{0} --> {1}]  ", printTime( seg.time.begin ), printTime( seg.time.end ) );
					foreach( sToken tok in res.getTokens( seg ) )
					{
						if( !args.print_special && tok.hasFlag( eTokenFlags.Special ) )
							continue;
						Console.Write( "{0}{1}{2}{3}", speaker, k_colors[ colorIndex( tok ) ], tok.text, "\x1B[0m" );
					}
					Console.WriteLine();
				}
				else
					Console.WriteLine( "[{0} --> {1}]  {2}{3}", printTime( seg.time.begin ), printTime( seg.time.end ), speaker, seg.text );
			}
		}
	}
}