#include "stdafx.h"
#include "../../Whisper/API/iContext.cl.h"
#include "TraceReader.h"
#include "../../Whisper/ML/testUtils.h"
#include "compare.h"
using namespace Tracing;
using namespace DirectCompute;

namespace
{
	inline const char* cstr( eItemType it )
	{
		switch( it )
		{
		case eItemType::Buffer: return "Buffer";
		case eItemType::Tensor: return "Tensor";
		}
		throw E_INVALIDARG;
	}
	inline const char* cstr( const CStringA& s ) { return s; }

	inline int tensorDims( __m128i vec )
	{
		const __m128i one = _mm_set1_epi32( 1 );
		const uint32_t bitmapOnes = (uint32_t)_mm_movemask_ps( _mm_castsi128_ps( _mm_cmpeq_epi32( vec, one ) ) );
		const uint32_t bitmapNotOnes = bitmapOnes ^ 0b1111u;
		unsigned long idx;
		if( !_BitScanReverse( &idx, bitmapNotOnes ) )
			return 0;
		return idx + 1;
	}

	int printSize( __m128i vec )
	{
		const int sz = tensorDims( vec );
		switch( sz )
		{
		case 0:
			printf( "[ scalar ]" );
			break;
		case 1:
			printf( "[ %i ]", _mm_cvtsi128_si32( vec ) );
			break;
		case 2:
			printf( "[ %i, %i ]", _mm_cvtsi128_si32( vec ), _mm_extract_epi32( vec, 1 ) );
			break;
		case 3:
			printf( "[ %i, %i, %i ]", _mm_cvtsi128_si32( vec ), _mm_extract_epi32( vec, 1 ), _mm_extract_epi32( vec, 2 ) );
			break;
		case 4:
			printf( "[ %i, %i, %i, %i ]", _mm_cvtsi128_si32( vec ), _mm_extract_epi32( vec, 1 ), _mm_extract_epi32( vec, 2 ), _mm_extract_epi32( vec, 3 ) );
			break;
		default:
			throw E_UNEXPECTED;
		}
		return sz;
	}

	class Comparer
	{
		TraceReader& readerA;
		TraceReader& readerB;

		bool diffBuffers( size_t i, const sTraceItem& a, const sTraceItem& b, const CStringA& name )
		{
			const size_t lenA = *(const uint64_t*)a.size.data();
			const size_t lenB = *(const uint64_t*)b.size.data();
			if( lenA != lenB )
			{
				printf( "Buffer %zu \"%s\": different size, %zu in trace A, %zu in trace B\n", i, cstr( name ), lenA, lenB );
				return false;
			}
			if( a.dataType != b.dataType )
			{
				printf( "Buffer %zu \"%s\": different data types\n", i, cstr( name ) );
				return false;
			}

			switch( a.dataType )
			{
			case eDataType::FP32:
				return buffersFp32( i, name, (const float*)readerA.payload( a ), (const float*)readerB.payload( b ), lenA );
			}
			throw E_NOTIMPL;
		}

		bool diffTensors( size_t i, const sTraceItem& a, const sTraceItem& b, const CStringA& name )
		{
			const __m128i ne1 = load( a.size );
			const __m128i ne2 = load( b.size );
			if( !vectorEqual( ne1, ne2 ) )
			{
				printf( "Tensor %zu \"%s\" - different size: trace A size is ", i, cstr( name ) );
				printSize( ne1 );
				printf( ", trace B size is " );
				printSize( ne2 );
				printf( "\n" );
				return false;
			}

			const __m128i stride1 = load( a.stride );
			const __m128i stride2 = load( b.stride );
			if( !vectorEqual( stride1, stride2 ) )
			{
				printf( "Tensor %zu \"%s\" - different memory layout\n", i, cstr( name ) );
				return false;
			}

			if( a.dataType != b.dataType )
			{
				printf( "Tensor %zu \"%s\": different data types\n", i, cstr( name ) );
				return false;
			}

			size_t elements = (uint32_t)_mm_cvtsi128_si32( ne1 );
			elements *= (uint32_t)_mm_extract_epi32( ne1, 1 );
			elements *= (uint32_t)_mm_extract_epi32( ne1, 2 );
			elements *= (uint32_t)_mm_extract_epi32( ne1, 3 );

			switch( a.dataType )
			{
			case eDataType::FP32:
				return tensorsFp32( i, name, (const float*)readerA.payload( a ), (const float*)readerB.payload( b ), elements, ne1, stride1 );
			}
			throw E_NOTIMPL;
		}

	protected:
		virtual bool buffersFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length ) = 0;
		virtual bool tensorsFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length, __m128i ne, __m128i nb ) = 0;

	public:

		Comparer( TraceReader& t1, TraceReader& t2 ) :
			readerA( t1 ), readerB( t2 ) { }

		bool compare( size_t i )
		{
			const sTraceItem& a = readerA[ i ];
			const sTraceItem& b = readerB[ i ];
			CStringA name1 = readerA.getName( a );
			CStringA name2 = readerB.getName( b );

			if( a.itemType != b.itemType )
			{
				printf( "Item %zu: different type, trace A %s \"%s\", trace B %s \"%s\"\n", i,
					cstr( a.itemType ), cstr( name1 ), cstr( b.itemType ), cstr( name2 ) );
				return false;
			}

			if( name1 != name2 )
			{
				printf( "%s %zu: different names, they are \"%s\" and \"%s\"\n", cstr( a.itemType ), i, cstr( name1 ), cstr( name2 ) );
				return false;
			}

			switch( a.itemType )
			{
			case eItemType::Buffer:
				return diffBuffers( i, a, b, name1 );
			case eItemType::Tensor:
				return diffTensors( i, a, b, name1 );
			default:
				throw E_INVALIDARG;
			}
		}
	};

	class PrintSummary : public Comparer
	{
		bool buffersFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length ) override;
		bool tensorsFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length, __m128i ne, __m128i nb ) override;

	public:
		PrintSummary( TraceReader& a, TraceReader& b ) : Comparer( a, b ) { }
	};

	bool PrintSummary::buffersFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length )
	{
		sTensorDiff diff = computeDiff( a, b, length );
		printf( "%s %zu \"%s\": ", cstr( eItemType::Buffer ), idx, cstr( name ) );
		diff.print();
		return true;
	}

	bool PrintSummary::tensorsFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length, __m128i ne, __m128i nb )
	{
		printSize( ne );
		printf( " " );
		sTensorDiff diff = computeDiff( a, b, length );
		printf( "%s %zu \"%s\": ", cstr( eItemType::Tensor ), idx, cstr( name ) );
		diff.print();
		return true;
	}

	class PrintDiff : public Comparer
	{
		bool buffersFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length ) override;
		bool tensorsFp32( size_t idx, const CStringA& name, const float* a, const float* b, size_t length, __m128i ne, __m128i nb ) override;
	public:
		PrintDiff( TraceReader& a, TraceReader& b ) : Comparer( a, b ) { }
	};

	bool PrintDiff::buffersFp32( size_t idx, const CStringA& name, const float* A, const float* B, size_t length )
	{
		printf( "idx\tA\tB\tA(hex)\tB(hex)\tdiff\n" );
		for( size_t i = 0; i < length; i++ )
		{
			const float a = *A;
			const float b = *B;
			__m128 vf = _mm_setr_ps( a, b, 0, 0 );
			__m128i vi = _mm_castps_si128( vf );
			const float diff = std::abs( a - b );
			printf( "%zu\t%g\t%g\t0x%08X\t0x%08X\t%g\n",
				i, a, b, _mm_cvtsi128_si32( vi ), _mm_extract_epi32( vi, 1 ), diff );
		}
		return true;
	}

	std::array<uint32_t, 4> storeSize( __m128i v )
	{
		std::array<uint32_t, 4> a;
		_mm_storeu_si128( ( __m128i* )a.data(), v );
		return a;
	}

	std::array<size_t, 4> storeStrides( __m128i v )
	{
		const __m128i zero = _mm_setzero_si128();
		std::array<size_t, 4> a;
		_mm_storeu_si128( ( __m128i* ) & a[ 0 ], _mm_unpacklo_epi32( v, zero ) );
		_mm_storeu_si128( ( __m128i* ) & a[ 2 ], _mm_unpackhi_epi32( v, zero ) );
		return a;
	}

	bool PrintDiff::tensorsFp32( size_t idx, const CStringA& name, const float* A, const float* B, size_t length, __m128i ne, __m128i nb )
	{
		const int dims = tensorDims( ne );
		const std::array<uint32_t, 4> size = storeSize( ne );
		const std::array<size_t, 4> strides = storeStrides( ne );
		CStringA line;
		if( dims > 4 )
			throw E_UNEXPECTED;

		for( int i = 0; i < dims; i++ )
		{
			const char c = "xyzw"[ i ];
			line.AppendChar( c );
			line.AppendChar( '\t' );
		}
		line += "A\tB\tA(hex)\tB(hex)\tdiff\n";
		printf( "%s", cstr( line ) );

		if( 0 == dims )
		{
			const float a = *A;
			const float b = *B;
			__m128 vf = _mm_setr_ps( a, b, 0, 0 );
			__m128i vi = _mm_castps_si128( vf );
			const float diff = std::abs( a - b );
			printf( "%g\t%g\t0x%08X\t0x%08X\t%g\n",
				a, b, _mm_cvtsi128_si32( vi ), _mm_extract_epi32( vi, 1 ), diff );
			return true;
		}

		size_t offLayer2 = 0;
		for( uint32_t w = 0; w < size[ 3 ]; w++, offLayer2 += strides[ 3 ] )
		{
			size_t offLayer = offLayer2;
			for( uint32_t z = 0; z < size[ 2 ]; z++, offLayer += strides[ 2 ] )
			{
				size_t offRow = offLayer;
				for( uint32_t y = 0; y < size[ 1 ]; y++, offRow += strides[ 1 ] )
				{
					size_t off = offRow;
					for( uint32_t x = 0; x < size[ 0 ]; x++, off += strides[ 0 ] )
					{
						line.Format( "%i\t", x );
						if( dims > 1 )
							line.AppendFormat( "%i\t", y );
						if( dims > 2 )
							line.AppendFormat( "%i\t", z );
						if( dims > 3 )
							line.AppendFormat( "%i\t", w );

						const float a = A[ off ];
						const float b = B[ off ];
						__m128 vf = _mm_setr_ps( a, b, 0, 0 );
						__m128i vi = _mm_castps_si128( vf );
						const float diff = std::abs( a - b );
						line.AppendFormat( "%g\t%g\t0x%08X\t0x%08X\t%g\n",
							a, b, _mm_cvtsi128_si32( vi ), _mm_extract_epi32( vi, 1 ), diff );
						printf( "%s", cstr( line ) );
					}
				}
			}
		}
		return true;
	}
}

HRESULT compareTraces( const CommandLineArgs& arguments )
{
	const wchar_t* pathA = arguments.inputs[ 0 ];
	const wchar_t* pathB = arguments.inputs[ 1 ];

	TraceReader a, b;
	HRESULT hr = a.open( pathA );
	if( FAILED( hr ) )
	{
		fwprintf( stderr, L"Unable to load trace A from \"%s\"", pathA );
		printError( hr );
		return hr;
	}

	hr = b.open( pathB );
	if( FAILED( hr ) )
	{
		fwprintf( stderr, L"Unable to load trace B from \"%s\"", pathA );
		printError( hr );
		return hr;
	}

	wprintf( L"Trace A:   %s\n", pathA );
	wprintf( L"Trace B:   %s\n", pathB );
	const size_t sizeA = a.size();
	const size_t sizeB = b.size();
	const size_t count = std::min( sizeA, sizeB );

	if( arguments.printDiff >= 0 )
	{
		if( arguments.printDiff >= (int64_t)count )
		{
			fprintf( stderr, "Trace A has %zu entries, trace B %zu entries; entry %zu ain't there\n",
				sizeA, sizeB, (size_t)arguments.printDiff );
			return E_INVALIDARG;
		}
		try
		{
			PrintDiff print{ a, b };
			print.compare( arguments.printDiff );
			return S_OK;
		}
		catch( HRESULT hr )
		{
			return hr;
		}
	}

	printf( "Trace A has %zu entries, trace B %zu entries, comparing first %zu\n", sizeA, sizeB, count );

	try
	{
		PrintSummary print{ a, b };
		for( size_t i = 0; i < count; i++ )
			if( !print.compare( i ) )
				return S_FALSE;
		return S_OK;
	}
	catch( HRESULT hr )
	{
		return hr;
	}
}