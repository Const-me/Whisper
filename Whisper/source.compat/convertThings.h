#pragma once
#include "../source/whisper.h"
#include "../API/sFullParams.h"
#include "../API/iTranscribeResult.cl.h"

Whisper::sFullParams makeNewParams( const whisper_full_params& rsi );

whisper_full_params makeOldParams( const Whisper::sFullParams& rsi, Whisper::iContext* context );

HRESULT makeNewResults( whisper_context* ctx, Whisper::eResultFlags flags, Whisper::iTranscribeResult** pp );