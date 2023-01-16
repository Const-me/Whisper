#include "stdafx.h"
#include "MappedResource.h"
using namespace DirectCompute;
#define CHECK( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) return __hr; }

MappedResource::MappedResource()
{
	mapped.pData = nullptr;
	mapped.RowPitch = mapped.DepthPitch = 0;
	resource = nullptr;
}

HRESULT MappedResource::map( ID3D11Resource* res, bool reading )
{
	if( nullptr == resource )
	{
		D3D11_MAP mt = reading ? D3D11_MAP_READ : D3D11_MAP_WRITE_DISCARD;
		CHECK( context()->Map( res, 0, mt, 0, &mapped ) );
		resource = res;
		return S_OK;
	}
	return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );
}

MappedResource::~MappedResource()
{
	if( nullptr != resource )
	{
		context()->Unmap( resource, 0 );
		resource = nullptr;
		mapped.pData = nullptr;
	}
}