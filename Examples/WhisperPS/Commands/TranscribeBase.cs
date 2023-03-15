using System;
using System.Management.Automation;

namespace Whisper
{
	public abstract class TranscribeBase: Cmdlet
	{
		[Parameter( Mandatory = false )]
		public string language { get; set; }

		[Parameter( Mandatory = false )]
		public SwitchParameter translate { get; set; }

		[Parameter( Mandatory = false )]
		public int threads { get; set; } = Environment.ProcessorCount;

		protected eLanguage languageCode { get; private set; } = eLanguage.English;

		protected void validateLanguage()
		{
			if( string.IsNullOrEmpty( language ) )
			{
				languageCode = eLanguage.English;
				return;
			}
			languageCode = Library.languageFromCode( language ) ?? throw new PSArgumentException( "language" );
		}

		protected void applyParams( ref Parameters parameters )
		{
			parameters.language = languageCode;
			parameters.cpuThreads = threads;
			if( translate )
				parameters.flags |= eFullParamsFlags.Translate;
			else
				parameters.flags &= ~eFullParamsFlags.Translate;
		}
	}
}