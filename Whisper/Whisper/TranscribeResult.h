#pragma once
#include "../API/iTranscribeResult.cl.h"
#include "../ComLightLib/comLightServer.h"

namespace Whisper
{
	class TranscribeResult : public ComLight::ObjectRoot<iTranscribeResult>
	{
		HRESULT COMLIGHTCALL getSize( sTranscribeLength& rdi ) const noexcept override final
		{
			rdi.countSegments = (uint32_t)segments.size();
			rdi.countTokens = (uint32_t)tokens.size();
			return S_OK;
		}
		const sSegment* COMLIGHTCALL getSegments() const noexcept override final
		{
			if( !segments.empty() )
				return segments.data();
			return nullptr;
		}
		const sToken* COMLIGHTCALL getTokens() const noexcept override final
		{
			if( !tokens.empty() )
				return tokens.data();
			return nullptr;
		}

	public:
		std::vector<sSegment> segments;
		std::vector<sToken> tokens;
		std::vector<std::string> segmentsText;
	};

	class TranscribeResultStatic : public ComLight::Object<TranscribeResult>
	{
		uint32_t COMLIGHTCALL Release() override final
		{
			// When the ref.counter reaches zero, Object.Release() method calls `delete this`.
			// We don't want that for the aggregated object.
			// Instead we only decrement the ref.counter, but do not delete the object even when the counter reaches zero.
			return RefCounter::implRelease();
		}
	};
}