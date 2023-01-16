#pragma once
#include "../ComLightLib/streams.h"
#include "../ComLightLib/comLightServer.h"
#define WIN32_LEAN_AND_MEAN
#include <atlfile.h>

class ReadStream : public ComLight::ObjectRoot<ComLight::iReadStream>
{
	CAtlFile file;
	// TODO: implement a buffer in this class, at least 256kb

	HRESULT COMLIGHTCALL read( void* lpBuffer, int nNumberOfBytesToRead, int& lpNumberOfBytesRead ) override final
	{
		return file.Read( lpBuffer, (DWORD)nNumberOfBytesToRead, *(DWORD*)&lpNumberOfBytesRead );
	}
	HRESULT COMLIGHTCALL seek( int64_t offset, ComLight::eSeekOrigin origin ) override final
	{
		return file.Seek( offset, (uint8_t)origin );
	}
	HRESULT COMLIGHTCALL getPosition( int64_t& position ) override final
	{
		return file.GetPosition( *(ULONGLONG*)&position );
	}
	HRESULT COMLIGHTCALL getLength( int64_t& length ) override final
	{
		return file.GetSize( *(ULONGLONG*)&length );
	}

public:

	HRESULT open( const wchar_t* path )
	{
		if( file )
			return HRESULT_CODE( ERROR_ALREADY_INITIALIZED );
		return file.Create( path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN );
	}
};