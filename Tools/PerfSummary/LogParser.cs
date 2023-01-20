using System.Globalization;
using System.Text.RegularExpressions;

namespace PerfSummary
{
	enum eInputClip: byte
	{
		jfk,
		columbia,
	}
	enum eWhisperModel: byte
	{
		medium,
		large
	}

	struct LogName
	{
		public readonly eInputClip clip;
		public readonly eWhisperModel model;
		public readonly string gpu;

		public override string ToString() => $"{clip}-{model}-{gpu}";

		public static LogName? tryParse( string path )
		{
			string? ext = Path.GetExtension( path );
			if( ext == null || !ext.Equals( ".txt", StringComparison.InvariantCultureIgnoreCase ) )
				return null;

			string name = Path.GetFileNameWithoutExtension( path );
			string[] fields = name.Split( '-' );
			if( fields.Length != 3 )
				return null;

			return new LogName( fields );
		}

		LogName( string[] fields )
		{
			clip = Enum.Parse<eInputClip>( fields[ 0 ] );
			model = Enum.Parse<eWhisperModel>( fields[ 1 ] );
			gpu = fields[ 2 ];
		}
	}

	record class LogData
	{
		public LogName name { get; init; }
		// The numbers are seconds
		public double runComplete { get; init; }
		public double encode { get; init; }
		public double decode { get; init; }
		// The numbers are megabytes
		public double ram { get; init; }
		public double vram { get; init; }
	}

	static class LogParser
	{
		public static IEnumerable<LogData> parse( string folder )
		{
			foreach( string path in Directory.EnumerateFiles( folder, "*.txt" ) )
			{
				LogName? name = LogName.tryParse( path );
				if( name == null )
					continue;
				yield return parseFile( name.Value, path );
			}
		}

		enum eSection: byte
		{
			CPU, GPU, Shaders, Memory
		}
		static readonly (string, eSection)[] sectionMarkers = new (string, eSection)[]
		{
			("CPU Tasks", eSection.CPU),
			("GPU Tasks", eSection.GPU),
			("Compute Shaders", eSection.Shaders),
			("Memory Usage", eSection.Memory),
		};

		static bool tryParseSection( ref eSection? section, string line )
		{
			foreach( (string marker, eSection s) in sectionMarkers )
			{
				if( line.Contains( marker ) )
				{
					section = s;
					return true;
				}
			}
			return false;
		}

		static bool tryParseTime( ref double? val, string key, string line )
		{
			if( !line.StartsWith( key ) )
				return false;
			if( !char.IsWhiteSpace( line[ key.Length ] ) )
				return false;

			line = line.Substring( key.Length ).TrimStart();
			int comma = line.IndexOf( ',' );
			if( comma > 0 )
				line = line.Substring( 0, comma );
			string[] fields = line.Split( ' ', StringSplitOptions.RemoveEmptyEntries );
			if( fields.Length != 2 )
				throw new ArgumentException();

			double v = double.Parse( fields[ 0 ], CultureInfo.InvariantCulture );
			switch( fields[ 1 ] )
			{
				case "seconds":
					val = v;
					return true;
				case "milliseconds":
					val = v / 1E3;
					return true;
				case "microseconds":
					val = v / 1E6;
					return true;
			}
			throw new ArgumentException();
		}

		static readonly Regex reMemory = new Regex( @"^Total\s+([0-9\.]+)\s+(\S+)\s+RAM, ([0-9\.]+)\s+(\S+)\s+VRAM$" );

		static double parseMemory( Match m, int iv )
		{
			double v = double.Parse( m.Groups[ iv ].Value, CultureInfo.InvariantCulture );
			string u = m.Groups[ iv + 1 ].Value;
			switch( u )
			{
				case "bytes":
					return v / ( 1 << 20 );
				case "KB":
					return v / ( 1 << 10 );
				case "MB":
					return v;
				case "GB":
					return v * ( 1 << 10 );
			}
			throw new ArgumentException();
		}

		static bool tryParseMemory( ref double? ram, ref double? vram, string line )
		{
			Match m = reMemory.Match( line );
			if( !m.Success )
				return false;
			ram = parseMemory( m, 1 );
			vram = parseMemory( m, 3 );
			return true;
		}

		static LogData parseFile( in LogName name, string path )
		{
			using var reader = File.OpenText( path );
			double? runComplete = null;
			double? encode = null;
			double? decode = null;
			double? ram = null;
			double? vram = null;
			eSection? section = null;

			while( true )
			{
				string? line = reader.ReadLine();
				if( line == null )
					break;
				if( string.IsNullOrEmpty( line ) )
					continue;
				if( tryParseSection( ref section, line ) )
					continue;

				switch( section )
				{
					case eSection.CPU:
						tryParseTime( ref runComplete, "RunComplete", line );
						break;
					case eSection.GPU:
						tryParseTime( ref encode, "Encode", line );
						tryParseTime( ref decode, "Decode", line );
						break;
					case eSection.Memory:
						tryParseMemory( ref ram, ref vram, line );
						break;
				}
			}

			if( null == runComplete || null == encode || null == decode || null == ram || null == vram )
				throw new ArgumentException();

			return new LogData()
			{
				name = name,
				runComplete = runComplete.Value,
				encode = encode.Value,
				decode = decode.Value,
				ram = ram.Value,
				vram = vram.Value,
			};
		}
	}
}