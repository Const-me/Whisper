#pragma once
#include "../../ComLightLib/streams.h"

namespace Whisper
{
	inline HRESULT readBytes( ComLight::iReadStream* stm, void* rdi, size_t cb )
	{
		if( cb > INT_MAX )
			return DISP_E_OVERFLOW;
		if( cb == 0 )
			return S_FALSE;
		int n;
		CHECK( stm->read( rdi, (int)cb, n ) );
		if( n != (int)cb )
			return E_EOF;
		return S_OK;
	}

	template<typename T>
	inline HRESULT readStruct( ComLight::iReadStream* stm, T& dest )
	{
		return readBytes( stm, &dest, sizeof( T ) );
	}
}