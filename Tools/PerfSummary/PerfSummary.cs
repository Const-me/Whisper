using System.Runtime.CompilerServices;

namespace PerfSummary
{
	internal class Program
	{
		static string getSolutionRoot( [CallerFilePath] string? path = null )
		{
			string? dir = Path.GetDirectoryName( path );
			dir = Path.GetDirectoryName( dir );
			dir = Path.GetDirectoryName( dir );
			return dir ?? throw new ApplicationException();
		}

		static void Main( string[] args )
		{
			string root = getSolutionRoot();
			root = Path.Combine( root, "SampleClips" );

			LogData[] logs = LogParser.parse( root )
				.OrderBy( x => x.name.clip )
				.ThenBy( x => x.name.model )
				.ThenBy( x => x.name.gpu )
				.ToArray();

			Summary.print( logs, root );
		}
	}
}