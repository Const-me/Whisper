#pragma once
#include "TranscribeStructs.h"

namespace Whisper
{
	__interface __declspec( novtable, uuid( "2871a73f-5ce3-48f8-8779-6582ee11935e" ) ) iTranscribeResult : public IUnknown
	{
		HRESULT __stdcall getSize( sTranscribeLength& rdi ) const;
		const sSegment* __stdcall getSegments() const;
		const sToken* __stdcall getTokens() const;
	};
}