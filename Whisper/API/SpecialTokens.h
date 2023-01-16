#pragma once

namespace Whisper
{
	struct SpecialTokens
	{
		// The end of a transcription, token_eot
		int TranscriptionEnd;
		// Start of a transcription, token_sot
		int TranscriptionStart;
		// Represents the previous word in the transcription. It is used to help the model predict the current word based on the context of the words that came before it.
		int PreviousWord;   // token_prev
		// Start of a sentence
		int SentenceStart;   // token_solm
		//Represents the word "not" in the transcription
		int Not;    // token_not
		//New transcription
		int TranscriptionBegin;    // token_beg

		// token_translate
		int TaskTranslate;
		// token_transcribe
		int TaskTranscribe;
	};
}