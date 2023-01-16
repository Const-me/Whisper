#include "stdafx.h"
#include "startup.h"
#include "device.h"

HRESULT DirectCompute::d3dStartup()
{
	HRESULT hr = DirectCompute::initialize();
	if( SUCCEEDED( hr ) )
		hr = createComputeShaders();
	return hr;
}

void DirectCompute::d3dShutdown()
{
	destroyComputeShaders();
	terminate();
}