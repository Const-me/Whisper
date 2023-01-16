#include "stdafx.h"
#include "CommandLineArgs.h"
#include <charconv>

static bool printUsage()
{
	fprintf( stderr, "Usage: compareTraces.exe trace1.bin trace2.bin [-diff N]\n" );
	return false;
}

bool CommandLineArgs::parse( int argc, wchar_t* argv[] )
{
	size_t idx = 0;
	CString sw;
	CStringA tmp;
	for( int i = 1; i < argc; i++ )
	{
		if( argv[ i ][ 0 ] != L'-' )
		{
			if( idx >= 2 )
				return printUsage();
			inputs[ idx ] = argv[ i ];
			idx++;
			continue;
		}
		sw = argv[ i ];
		if( 0 == sw.CompareNoCase( L"-diff" ) )
		{
			i++;
			if( i >= argc )
				return printUsage();
			tmp.Format( "%S", argv[ i ] );
			tmp.Trim();
			uint64_t v;
			auto res = std::from_chars( tmp, cstr( tmp ) + tmp.GetLength(), v );
			if( res.ec != (std::errc)0 )
			{
				fprintf( stderr, "Unable to parse string into number\n" );
				return false;
			}
			printDiff = v;
			continue;
		}
		return printUsage();
	}

	if( idx != 2 )
		return printUsage();

	return true;
}