using System;
using System.IO;
using System.Text;
using Whisper.Internal;

namespace Whisper
{
	/// <summary>
	/// <para type="synopsis">This object holds the results of a transcription.</para>
	/// <para type="description">It retains all data produced by the language model, including tokens and probabilities</para>
	/// </summary>
	public sealed class Transcription: IDisposable
	{
		iTranscribeResult result;

		internal Transcription( string src, iTranscribeResult result )
		{
			Source = src;
			this.result = result;
		}

		public void Dispose()
		{
			result?.Dispose();
			result = null;
			GC.SuppressFinalize( this );
		}

		~Transcription()
		{
			result?.Dispose();
			result = null;
		}

		internal TranscribeResult getResult() =>
			new TranscribeResult( result );

		string makeText()
		{
			TranscribeResult res = getResult();
			StringBuilder sb = new StringBuilder();
			bool first = true;
			foreach( var seg in res.segments )
			{
				if( first )
					first = false;
				else
					sb.AppendLine();
				sb.Append( seg.text.Trim() );
			}
			return sb.ToString();
		}

		/// <summary>Return the text</summary>
		public override string ToString() => makeText();

		/// <summary>Source file</summary>
		public string Source { get; }

		/// <summary>Source file name, no extension</summary>
		public string SourceName =>
			Path.GetFileNameWithoutExtension( Source );

		/// <summary>Text of the complete transcription</summary>
		public string Text => makeText();
	}
}