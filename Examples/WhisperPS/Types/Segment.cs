using System;

namespace Whisper
{
	/// <summary>One text segment of a transcription</summary>
	public sealed class Segment
	{
		internal Segment( in sSegment seg )
		{
			Begin = seg.time.begin;
			End = seg.time.end;
			Text = seg.text.Trim();
		}

		/// <summary>Timestamp of the start of the segment, since the start of the media</summary>
		public TimeSpan Begin { get; }
		/// <summary>Timestamp of the end of the segment, since the start of the media</summary>
		public TimeSpan End { get; }

		/// <summary>Text of the segment</summary>
		public string Text { get; }

		/// <summary>A string representation of this object</summary>
		public override string ToString() => Text;
	}
}