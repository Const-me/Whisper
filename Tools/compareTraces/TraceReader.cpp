#include "stdafx.h"
#include "TraceReader.h"
using namespace Tracing;

const sTraceItem& TraceReader::operator[]( size_t idx ) const
{
	if( idx >= countItems )
		throw E_BOUNDS;
	return items[ idx ];
}

CStringA TraceReader::getName( const sTraceItem& item ) const
{
	const size_t idx = item.stringIndex;
	if( idx >= countStrings )
		throw E_BOUNDS;
	const char* const source = stringData + stringIndex[ idx ];
	CStringA res;
	res.Format( source, item.formatArgs[ 0 ], item.formatArgs[ 1 ], item.formatArgs[ 2 ], item.formatArgs[ 3 ] );
	return res;
}

HRESULT TraceReader::open( LPCTSTR path )
{
	CHECK( file.Create( path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING ) );
	CHECK( mapping.MapFile( file ) );

	const uint8_t* rsi = mapping;
	const sFileHeader& header = *(const sFileHeader*)rsi;
	if( header.magic != header.correctMagic )
		return E_INVALIDARG;
	countItems = header.countItems;
	countStrings = header.countStrings;

	rsi += sizeof( sFileHeader );
	payloadPointer = rsi;

	rsi += header.bytesPayload;
	stringIndex = (const uint32_t*)( rsi );
	stringData = (const char*)( rsi + countStrings * 4 );

	rsi += header.bytesStrings;
	items = (const sTraceItem*)rsi;

	return S_OK;
}