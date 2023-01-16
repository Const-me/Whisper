#pragma once
#include "../API/iContext.cl.h"
#include "../ComLightLib/comLightServer.h"
#include "WhisperModel.h"
#include "../ComLightLib/streams.h"

namespace Whisper
{
	using ComLight::iReadStream;

	class ModelImpl : public ComLight::ObjectRoot<iModel>
	{
		WhisperModel model;

		HRESULT COMLIGHTCALL createContext( iContext** pp ) override final;

		HRESULT COMLIGHTCALL getSpecialTokens( SpecialTokens& rdi ) override final
		{
			model.vocab.getSpecialTokens( rdi );
			return S_OK;
		}

		HRESULT COMLIGHTCALL isMultilingual() override final
		{
			return model.vocab.is_multilingual() ? S_OK : S_FALSE;
		}

		const char* COMLIGHTCALL stringFromToken( whisper_token token ) override final
		{
			return model.vocab.string( token );
		}

	public:

		HRESULT FinalConstruct();
		void FinalRelease();

		HRESULT load( iReadStream* stm, bool hybrid, const sLoadModelCallbacks* callbacks );
	};
}