using Microsoft.PowerShell.Commands;
using System;
using System.IO;
using System.Management.Automation;
using System.Management.Automation.Runspaces;

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

		static bool isFullPath( string path )
		{
			if( string.IsNullOrWhiteSpace( path ) || path.IndexOfAny( Path.GetInvalidPathChars() ) != -1 || !Path.IsPathRooted( path ) )
				return false;

			string pathRoot = Path.GetPathRoot( path );
			if( pathRoot.Length <= 2 && pathRoot != "/" ) // Accepts X:\ and \\UNC\PATH, rejects empty string, \ and X:, but accepts / to support Linux
				return false;

			if( pathRoot[ 0 ] != '\\' || pathRoot[ 1 ] != '\\' )
				return true; // Rooted and not a UNC path

			return pathRoot.Trim( '\\' ).IndexOf( '\\' ) != -1; // A UNC server name without a share name (e.g "\\NAME" or "\\NAME\") is invalid
		}

		static string currentDirectory( this PSCmdlet cmd )
		{
			PathInfo pi = cmd.SessionState.Path.CurrentFileSystemLocation;
			if( pi.Provider.ImplementingType != typeof( FileSystemProvider ) )
				throw new ArgumentException( "Relative paths only work for file systems" );
			return pi.ProviderPath;
		}

		public static string absolutePath( this PSCmdlet cmd, string path )
		{
			if( isFullPath( path ) )
				return path;
			string dir = cmd.currentDirectory();
			return Path.GetFullPath( Path.Combine( dir, path ) );
		}
	}
}