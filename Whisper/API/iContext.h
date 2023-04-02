#pragma once
#include "iTranscribeResult.h"
#include "SpecialTokens.h"
#include "loggerApi.h"
#include "sLanguageList.h"
#include "sLoadModelCallbacks.h"
#include "sModelSetup.h"

namespace Whisper
{
	__interface iModel;
	__interface iAudioBuffer;
	__interface iAudioReader;
	__interface iAudioCapture;
	struct sCaptureCallbacks;
	struct sFullParams;
	enum struct eModelImplementation : uint32_t;
	enum struct eSamplingStrategy : int;
	using whisper_token = int;
	struct sProgressSink;

	__interface __declspec( novtable, uuid( "b9956374-3b18-4943-90f2-2ab18a404537" ) ) iContext : public IUnknown
	{
		// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
		// Uses the specified decoding strategy to obtain the text.
		HRESULT __stdcall runFull( const sFullParams& params, const iAudioBuffer* buffer );
		HRESULT __stdcall runStreamed( const sFullParams& params, const sProgressSink& progress, const iAudioReader* reader );
		HRESULT __stdcall runCapture( const sFullParams& params, const sCaptureCallbacks& callbacks, const iAudioCapture* reader );

		HRESULT __stdcall getResults( eResultFlags flags, iTranscribeResult** pp ) const;
		// Try to detect speaker by comparing channels of the stereo PCM data
		HRESULT __stdcall detectSpeaker( const sTimeInterval& time, eSpeakerChannel& result ) const;

		HRESULT __stdcall getModel( iModel** pp );

		HRESULT __stdcall fullDefaultParams( eSamplingStrategy strategy, sFullParams* rdi );

		// Performance information
		HRESULT __stdcall timingsPrint();
		HRESULT __stdcall timingsReset();
	};

	__interface __declspec( novtable, uuid( "abefb4c9-e8d8-46a3-8747-5afbadef1adb" ) ) iModel : public IUnknown
	{
		HRESULT __stdcall createContext( iContext** pp );

		HRESULT __stdcall tokenize( const char* text, pfnDecodedTokens pfn, void* pv );

		HRESULT __stdcall isMultilingual();

		HRESULT __stdcall getSpecialTokens( SpecialTokens& rdi );

		// Token Id -> String
		const char* __stdcall stringFromToken( whisper_token token );

		HRESULT __stdcall clone( iModel** rdi );
	};

	HRESULT __stdcall setupLogger( const sLoggerSetup& setup );
	HRESULT __stdcall loadModel( const wchar_t* path, const sModelSetup& setup, const sLoadModelCallbacks* callbacks, iModel** pp );

	uint32_t __stdcall findLanguageKeyW( const wchar_t* lang );
	uint32_t __stdcall findLanguageKeyA( const char* lang );

	HRESULT __stdcall getSupportedLanguages( sLanguageList& rdi );

	HRESULT __stdcall listGPUs( pfnListAdapters pfn, void* pv );
}

#include "sFullParams.h"