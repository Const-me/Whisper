#pragma once
#include "TranscribeStructs.h"
#include "../../ComLightLib/comLightCommon.h"

namespace Whisper
{
	struct iTranscribeResult : public ComLight::IUnknown
	{
		DEFINE_INTERFACE_ID( "{2871a73f-5ce3-48f8-8779-6582ee11935e}" );

		virtual HRESULT COMLIGHTCALL getSize( sTranscribeLength& rdi ) const = 0;
		virtual const sSegment* COMLIGHTCALL getSegments() const = 0;
		virtual const sToken* COMLIGHTCALL getTokens() const = 0;
	};
}