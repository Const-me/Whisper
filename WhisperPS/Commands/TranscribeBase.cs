using System;
using System.Management.Automation;
using Whisper.Internal;

namespace Whisper
{
	/// <summary>Base class for transcribing cmdlets, it contains a few common parameters</summary>
	public abstract class TranscribeBase: PSCmdlet
	{
		const eLanguage defaultLanguage = eLanguage.English;

		/// <summary>
		/// <para type="synopsis">Whisper model in VRAM</para>
		/// <para type="description">Use <c>Import-WhisperModel</c> command to load the model from disk</para>
		/// </summary>
		[Parameter( Mandatory = true, Position = 0 )]
		public Model model { get; set; }

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

		/// <summary>
		/// <para type="synopsis">Optional initial prompt for the model. For example, &quot;繁體中文&quot; for traditional Chinese, &quot;简体中文&quot; for simplified.</para>
		/// </summary>
		[Parameter( Mandatory = false )]
		public string prompt { get; set; }

		/// <summary>Convert the provided text into tokens</summary>
		internal static int[] tokenize( iModel model, string text )
		{
			int[] result = null;
			pfnDecodedTokens pfn = delegate ( int[] arr, int length, IntPtr pv )
			{
				result = arr;
			};
			model.tokenize( text, pfn, IntPtr.Zero );
			return result;
		}

		/// <summary>
		/// <para type="synopsis">Maximum segment length in characters</para>
		/// <para type="description">The default is 60</para>
		/// </summary>
		[Parameter( Mandatory = false )]
		public ushort maxSegmentLength { get; set; } = 60;

		internal eLanguage languageCode { get; private set; } = eLanguage.English;

		static eLanguage parseLanguage( string lang )
		{
			// When no parameter supplied, default to English
			if( string.IsNullOrEmpty( lang ) )
				return defaultLanguage;

			// Try human-readable names, such as "Chinese" or "Ukrainian"
			eLanguage res;
			const bool ignoreCase = true;
			if( Enum.TryParse( lang, ignoreCase, out res ) )
				return res;

			// Try OpenAI's language codes, the 1988 version of ISO 639-1
			eLanguage? nullable = Library.languageFromCode( lang );
			if( nullable.HasValue )
				return nullable.Value;

			throw new PSArgumentException( $"Unable to parse the string \"{lang}\" into language" );
		}

		internal void validateLanguage()
		{
			languageCode = parseLanguage( language );
			if( languageCode == eLanguage.English && translate )
				throw new ArgumentException( "The translate feature translates speech to English.\nIt’s not available when the audio language is already English.", nameof( language ) );
		}

		internal void applyParams( ref Parameters parameters )
		{
			parameters.language = languageCode;
			parameters.cpuThreads = threads;
			if( translate )
				parameters.flags |= eFullParamsFlags.Translate;
			else
				parameters.flags &= ~eFullParamsFlags.Translate;
			parameters.max_len = maxSegmentLength;
		}

		internal sProgressSink makeProgressSink( string what )
		{
			sProgressSink res = default;

			res.pfn = delegate ( double progressValue, IntPtr context, IntPtr pv )
			{
				try
				{
					this.writeProgress( progressValue, what );
					return 0;
				}
				catch( Exception ex )
				{
					return ex.HResult;
				}
			};

			return res;
		}
	}
}