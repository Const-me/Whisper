using System;
using System.Text;
using Whisper.Internal;

namespace Whisper
{
	public sealed class Transcription: IDisposable
	{
		iTranscribeResult result;

		public Transcription( iTranscribeResult result )
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

		public override string ToString()
		{
			TranscribeResult res = new TranscribeResult( result );
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