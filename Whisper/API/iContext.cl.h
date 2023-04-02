#pragma once
#include "../../ComLightLib/comLightCommon.h"
#include "iTranscribeResult.cl.h"
#include "SpecialTokens.h"
#include "loggerApi.h"
#include "sLanguageList.h"
#include "sLoadModelCallbacks.h"
#include "sModelSetup.h"

namespace Whisper
{
	struct iModel;
	struct iAudioBuffer;
	struct iAudioReader;
	struct iAudioCapture;
	struct sCaptureCallbacks;
	struct sFullParams;
	enum struct eModelImplementation : uint32_t;
	enum struct eSamplingStrategy : int;
	using whisper_token = int;
	struct sProgressSink;

	struct DECLSPEC_NOVTABLE iContext : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "{b9956374-3b18-4943-90f2-2ab18a404537}" );

		// Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
		// Uses the specified decoding strategy to obtain the text.
		virtual HRESULT COMLIGHTCALL runFull( const sFullParams& params, const iAudioBuffer* buffer ) = 0;
		virtual HRESULT COMLIGHTCALL runStreamed( const sFullParams& params, const sProgressSink& progress, const iAudioReader* reader ) = 0;
		virtual HRESULT COMLIGHTCALL runCapture( const sFullParams& params, const sCaptureCallbacks& callbacks, const iAudioCapture* reader ) = 0;

		virtual HRESULT COMLIGHTCALL getResults( eResultFlags flags, iTranscribeResult** pp ) const = 0;
		// Try to detect speaker by comparing channels of the stereo PCM data
		virtual HRESULT COMLIGHTCALL detectSpeaker( const sTimeInterval& time, eSpeakerChannel& result ) const = 0;

		virtual HRESULT COMLIGHTCALL getModel( iModel** pp ) = 0;

		virtual HRESULT COMLIGHTCALL fullDefaultParams( eSamplingStrategy strategy, sFullParams* rdi ) = 0;

		// Performance information
		virtual HRESULT COMLIGHTCALL timingsPrint() = 0;
		virtual HRESULT COMLIGHTCALL timingsReset() = 0;
	};

	struct DECLSPEC_NOVTABLE iModel : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "{abefb4c9-e8d8-46a3-8747-5afbadef1adb}" );

		virtual HRESULT COMLIGHTCALL createContext( iContext** pp ) = 0;

		virtual HRESULT COMLIGHTCALL tokenize( const char* text, pfnDecodedTokens pfn, void* pv ) = 0;
		virtual HRESULT COMLIGHTCALL isMultilingual() = 0;
		virtual HRESULT COMLIGHTCALL getSpecialTokens( SpecialTokens& rdi ) = 0;

		// Token Id -> String
		virtual const char* COMLIGHTCALL stringFromToken( whisper_token token ) = 0;

		virtual HRESULT COMLIGHTCALL clone( iModel** rdi ) = 0;
	};

	HRESULT COMLIGHTCALL setupLogger( const sLoggerSetup& setup );
	HRESULT COMLIGHTCALL loadModel( const wchar_t* path, const sModelSetup& setup, const sLoadModelCallbacks* callbacks, iModel** pp );

	uint32_t COMLIGHTCALL findLanguageKeyW( const wchar_t* lang );
	uint32_t COMLIGHTCALL findLanguageKeyA( const char* lang );

	HRESULT COMLIGHTCALL getSupportedLanguages( sLanguageList& rdi );

	HRESULT COMLIGHTCALL listGPUs( pfnListAdapters pfn, void* pv );
}

#include "sFullParams.h"