#include "stdafx.h"
#include "startup.h"
#include "device.h"

HRESULT DirectCompute::d3dStartup( uint32_t flags )
{
	HRESULT hr = DirectCompute::initialize( flags );
	if( SUCCEEDED( hr ) )
		hr = createComputeShaders();
	return hr;
}

void DirectCompute::d3dShutdown()
{
	destroyComputeShaders();
	terminate();
}