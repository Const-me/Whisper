// Missing XML comment for publicly visible type or member
// TODO: remove this line and document them.
#pragma warning disable CS1591

using System;

namespace Whisper
{
	/// <summary>Available sampling strategies</summary>
	public enum eSamplingStrategy: int
	{
		/// <summary>Always select the most probable token</summary>
		Greedy,
		/// <summary>TODO: not implemented yet!</summary>
		BeamSearch,
	};

	[Flags]
	public enum eFullParamsFlags: uint
	{
		None = 0,
		Translate = 1,
		NoContext = 2,
		SingleSegment = 4,
		PrintSpecial = 8,
		PrintProgress = 0x10,
		PrintRealtime = 0x20,
		PrintTimestamps = 0x40,

		// Experimental
		TokenTimestamps = 0x100,
		SpeedupAudio = 0x200,
	};

	/// <summary>Transcribe parameters</summary>
	public struct Parameters
	{
		/// <summary>Sampling strategy</summary>
		public eSamplingStrategy strategy;

		/// <summary>Count of CPU worker threads to use</summary>
		/// <remarks>So far, the GPU model only uses CPU threads for MEL spectrograms</remarks>
		public int cpuThreads;

		public int n_max_text_ctx;
		/// <summary>start offset in ms</summary>
		public int offset_ms;
		/// <summary>audio duration to process in ms</summary>
		public int duration_ms;
		public eFullParamsFlags flags;

		/// <summary>Set or clear the specified flag in the <see cref="flags" /> field of this structure</summary>
		public void setFlag( eFullParamsFlags flag, bool set )
		{
			if( flag != eFullParamsFlags.None )
			{
				if( set )
					flags |= flag;
				else
					flags &= ~flag;
				return;
			}
			throw new ArgumentException();
		}

		/// <summary>Language</summary>
		public eLanguage language;

		// [EXPERIMENTAL] token-level timestamps
		/// <summary>timestamp token probability threshold (~0.01)</summary>
		public float thold_pt;
		/// <summary>timestamp token sum probability threshold (~0.01)</summary>
		public float thold_ptsum;
		/// <summary>max segment length in characters</summary>
		public int max_len;
		/// <summary>max tokens per segment (0 = no limit)</summary>
		public int max_tokens;

		public struct sGreedy
		{
			public int n_past;
		}
		public sGreedy greedy;

		public struct sBeamSearch
		{
			public int n_past;
			public int beam_width;
			public int n_best;
		}
		public sBeamSearch beamSearch;

		// [EXPERIMENTAL] speed-up techniques
		/// <summary>overwrite the audio context size (0 = use default)</summary>
		public int audioContextSize;
	}
}