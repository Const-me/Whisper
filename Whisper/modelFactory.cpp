#include "stdafx.h"
#include "modelFactory.h"
#include "API/iContext.cl.h"

HRESULT COMLIGHTCALL Whisper::loadModel( const wchar_t* path, eModelImplementation impl, const sLoadModelCallbacks* callbacks, iModel** pp )
{
	switch( impl )
	{
	case eModelImplementation::GPU:
		return loadGpuModel( path, false, callbacks, pp );
	case eModelImplementation::Hybrid:
		return loadGpuModel( path, true, callbacks, pp );
	case eModelImplementation::Reference:
		return loadReferenceCpuModel( path, pp );
	}

	logError( u8"Unknown model implementation 0x%X", (int)impl );
	return E_INVALIDARG;
}