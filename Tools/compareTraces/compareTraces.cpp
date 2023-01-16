#include "stdafx.h"
#include <stdio.h>
#include "compare.h"
#include "CommandLineArgs.h"

int wmain( int argc, wchar_t* argv[] )
{
	CommandLineArgs cla;
	if( !cla.parse( argc, argv ) )
		return 1;

	HRESULT hr = compareTraces( cla );
	if( SUCCEEDED( hr ) )
		return 0;
	return hr;
}