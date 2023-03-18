using System;
using System.Text;

namespace Whisper.Internal
{
	static class MarshalEx
	{
		/// <summary>Workaround for the missing <c>Marshal.PtrToStringUTF8</c> method</summary>
		public static string PtrToStringUTF8( IntPtr ptr )
		{
			if( ptr != IntPtr.Zero )
			{
				unsafe
				{
					byte* stringStart = (byte*)ptr;
					byte* stringEnd = stringStart;
					while( true )
					{
						if( 0 == *stringEnd )
							break;
						stringEnd++;
					}

					int len = (int)( stringEnd - stringStart );
					return Encoding.UTF8.GetString( stringStart, len );
				}
			}
			return null;
		}
	}
}