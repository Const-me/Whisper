using System;
using System.Management.Automation;

namespace Whisper
{
	/// <summary>Base class for transcribing cmdlets, it contains a few common parameters</summary>
	public abstract class TranscribeBase: Cmdlet
	{
		/// <summary>
		/// <para type="synopsis">Language code to use</para>
		/// </summary>
		[Parameter( Mandatory = false )]
		public string language { get; set; }

		/// <summary>
		/// <para type="synopsis">Specify this switch to translate the language into English</para>
		/// </summary>
		[Parameter( Mandatory = false )]
		public SwitchParameter translate { get; set; }

		/// <summary>
		/// <para type="synopsis">Count of CPU threads to use</para>
		/// <para type="description">Specify 1 to disable multithreading</para>
		/// </summary>
		[Parameter( Mandatory = false )]
		public int threads { get; set; } = Environment.ProcessorCount;

		/// <summary></summary>
		protected eLanguage languageCode { get; private set; } = eLanguage.English;

		protected void validateLanguage()
		{
			if( string.IsNullOrEmpty( language ) )
				languageCode = eLanguage.English;
			else
				languageCode = Library.languageFromCode( language ) ?? throw new PSArgumentException( "language" );

			if( languageCode == eLanguage.English && translate )
				throw new ArgumentException( "The translate feature translates speech to English.\nIt’s not available when the audio language is already English.", nameof( language ) );
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