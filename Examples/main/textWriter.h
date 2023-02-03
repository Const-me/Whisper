#pragma once
#include "../../Whisper/API/iContext.cl.h"

// These functions print output segments into text files of various formats
HRESULT writeText( Whisper::iContext* context, LPCTSTR audioPath, bool timestamps );
HRESULT writeSubRip( Whisper::iContext* context, LPCTSTR audioPath );
HRESULT writeWebVTT( Whisper::iContext* context, LPCTSTR audioPath );