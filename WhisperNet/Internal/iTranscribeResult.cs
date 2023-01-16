#pragma warning disable CS0649 // Field is never assigned to
using ComLight;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace Whisper.Internal
{
	/// <summary>Size of the buffers owned by the <see cref="iTranscribeResult" /> object</summary>
	public readonly struct sTranscribeLength
	{
		/// <summary>Count of segments</summary>
		public readonly int countSegments;
		/// <summary>Total count of tokens, for all segments combined</summary>
		public readonly int countTokens;
	}

	/// <summary>Output data from the model</summary>
	[ComInterface( "2871a73f-5ce3-48f8-8779-6582ee11935e", eMarshalDirection.ToManaged ), CustomConventions( typeof( NativeLogger ) )]
	public interface iTranscribeResult
	{
		/// <summary>Get size of the buffers</summary>
		[RetValIndex, EditorBrowsable( EditorBrowsableState.Never )]
		public sTranscribeLength getSize();

		/// <summary>Pointer to segment data, a vector of <see cref="sSegment" /> structures</summary>
		[EditorBrowsable( EditorBrowsableState.Never )]
		public IntPtr getSegments();

		/// <summary>Pointer to tokens data, a vector of <see cref="sToken" /> structures</summary>
		[EditorBrowsable( EditorBrowsableState.Never )]
		public IntPtr getTokens();
	}
}

namespace Whisper
{
	/// <summary>Start and end times of a segment or token</summary>
	/// <remarks>The times are relative to the start of the media</remarks>
	public readonly struct sTimeInterval
	{
		/// <summary>Start time</summary>
		public readonly TimeSpan begin;
		/// <summary>End time</summary>
		public readonly TimeSpan end;
	}

	/// <summary>Segment data</summary>
	public readonly struct sSegment
	{
		internal readonly IntPtr m_text;
		/// <summary>Segment text</summary>
		public string? text => Marshal.PtrToStringUTF8( m_text );
		/// <summary>Start and end times of the segment</summary>
		public readonly sTimeInterval time;
		/// <summary>Slice of the tokens</summary>
		public readonly int firstToken, countTokens;
	}

	/// <summary>Token flags</summary>
	[Flags]
	public enum eTokenFlags: uint
	{
		/// <summary>The token is special</summary>
		Special = 1,
	}

	/// <summary>Token data</summary>
	public readonly struct sToken
	{
		internal readonly IntPtr m_text;
		/// <summary>Token text</summary>
		public string? text => Marshal.PtrToStringUTF8( m_text );
		/// <summary>Start and end times of the token</summary>
		public readonly sTimeInterval time;
		/// <summary>Probability of the token</summary>
		public readonly float probability;
		/// <summary>Probability of the timestamp token</summary>
		public readonly float probabilityTimestamp;
		/// <summary>Sum of probabilities of all timestamp tokens</summary>
		public readonly float ptsum;
		/// <summary>Voice length of the token</summary>
		public readonly float vlen;
		/// <summary>Token id</summary>
		public readonly int id;
		/// <summary>Token flags</summary>
		readonly eTokenFlags flags;
		/// <summary>True if the token flags has the specified bit set</summary>
		public bool hasFlag( eTokenFlags bit ) => flags.HasFlag( bit );
	}

	/// <summary>Output data from the model</summary>
	public readonly ref struct TranscribeResult
	{
		/// <summary>Segments in the results</summary>
		public readonly ReadOnlySpan<sSegment> segments;
		/// <summary>Tokens in the results, for all segments</summary>
		public readonly ReadOnlySpan<sToken> tokens;

		internal TranscribeResult( Internal.iTranscribeResult i )
		{
			Internal.sTranscribeLength len = i.getSize();
			unsafe
			{
				// This does not copy the buffers to managed memory.
				// Instead, the C# spans directly reference the native memory stored in these std::vectors
				if( len.countSegments > 0 )
					segments = new ReadOnlySpan<sSegment>( (void*)i.getSegments(), len.countSegments );
				else
					segments = ReadOnlySpan<sSegment>.Empty;

				if( len.countTokens > 0 )
					tokens = new ReadOnlySpan<sToken>( (void*)i.getTokens(), len.countTokens );
				else
					tokens = ReadOnlySpan<sToken>.Empty;
			}
		}

		/// <summary>Get tokens for the specified segment</summary>
		public ReadOnlySpan<sToken> getTokens( in sSegment seg ) =>
			tokens.Slice( seg.firstToken, seg.countTokens );
	}
}