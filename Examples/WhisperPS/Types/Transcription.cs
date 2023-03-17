using System;
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

		internal Transcription( iTranscribeResult result )
		{
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

		public override string ToString()
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
				sb.Append( seg.text );
			}

			return sb.ToString();
		}
	}
}