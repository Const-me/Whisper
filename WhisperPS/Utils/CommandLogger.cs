using System;
using System.Management.Automation;

namespace Whisper
{
	static class CommandLogger
	{
		static CommandLogger()
		{
			Library.setLogSink( eLogLevel.Debug, eLoggerFlags.SkipFormatMessage, sink );
		}

		static void sink( eLogLevel level, string message )
		{
			switch( level )
			{
				case eLogLevel.Warning:
					cmdlet?.WriteWarning( message );
					break;
				case eLogLevel.Info:
					cmdlet?.WriteInformation( message, null );
					break;
				case eLogLevel.Debug:
					cmdlet?.WriteDebug( message );
					break;
				// Errors usually become C# exceptions
			}
		}

		[ThreadStatic]
		static Cmdlet cmdlet;

		sealed class Impl: IDisposable
		{
			public Impl( Cmdlet c )
			{
				cmdlet = c;
			}

			void IDisposable.Dispose()
			{
				cmdlet = null;
			}
		}

		public static IDisposable setupLog( this Cmdlet cmd ) =>
			new Impl( cmd );
	}
}