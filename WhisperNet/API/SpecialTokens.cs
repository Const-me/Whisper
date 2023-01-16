namespace Whisper
{
	/// <summary>Special tokens defined in the model</summary>
	public readonly struct SpecialTokens
	{
		/// <summary>The end of a transcription</summary>
		public readonly int TranscriptionEnd; // token_eot
		/// <summary>Start of a transcription</summary>
		public readonly int TranscriptionStart;   // token_sot
		/// <summary>Represents the previous word in the transcription. It is used to help the model predict the current word based on the context of the words that came before it.</summary>
		public readonly int PreviousWord;   // token_prev
		/// <summary>Start of a sentence</summary>
		public readonly int SentenceStart;   // token_solm
		/// <summary>Represents the word "not" in the transcription</summary>
		public readonly int Not;    // token_not
		/// <summary>New transcription</summary>
		public readonly int TranscriptionBegin;    // token_beg
		/// <summary>token_translate</summary>
		public readonly int TaskTranslate;
		/// <summary>token_transcribe</summary>
		public readonly int TaskTranscribe;
	}
}