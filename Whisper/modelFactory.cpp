#include "stdafx.h"
#include "modelFactory.h"
#include "API/iContext.cl.h"

HRESULT COMLIGHTCALL Whisper::loadModel( const wchar_t* path, eModelImplementation impl, uint32_t flags, const sLoadModelCallbacks* callbacks, iModel** pp )
{
	switch( impl )
	{
	case eModelImplementation::GPU:
		return loadGpuModel( path, false, flags, callbacks, pp );
	case eModelImplementation::Hybrid:
		return loadGpuModel( path, true, flags, callbacks, pp );
	case eModelImplementation::Reference:
		if( 0 != flags )
			logWarning( u8"The reference model doesn’t currently use any flags, argument ignored" );
		return loadReferenceCpuModel( path, pp );
	}

	logError( u8"Unknown model implementation 0x%X", (int)impl );
	return E_INVALIDARG;
}