#include "stdafx.h"
#include "TraceWriter.h"
#include <atlfile.h>
#include <atlcoll.h>
#include <atlstr.h>
#include "TraceStructures.h"
#include "../../ML/Tensor.h"
#include "../../CPU/Tensor.h"
#include <Shlobj.h>
using namespace Tracing;

namespace
{
	static HRESULT createDir( LPCTSTR pathFile )
	{
		LPCWSTR fn = PathFindFileName( pathFile );
		if( fn == pathFile )
			return E_FAIL;

		const int cc = (int)( fn - pathFile );
		CString dir{ pathFile, cc };
		if( PathIsDirectory( dir ) )
			return S_OK;
		const int status = SHCreateDirectoryEx( nullptr, dir, nullptr );
		if( 0 == status )
			return S_OK;
		return HRESULT_FROM_WIN32( status );
	}

	class TraceFileWriter
	{
		CAtlFile file;
		// Concatenated strings, including the 0 terminators
		std::vector<char> stringsData;
		// Index = string ID, value = start offset into stringsData
		std::vector<uint32_t> stringsIndex;
		// Hash map to unduplicate these strings
		CAtlMap<CStringA, uint32_t> stringsHash;

		uint32_t addString( const CStringA& s )
		{
			auto p = stringsHash.Lookup( s );
			if( p != nullptr )
				return p->m_value;

			const uint32_t off = (uint32_t)stringsData.size();
			const char* rsi = s;
			stringsData.insert( stringsData.end(), rsi, rsi + s.GetLength() + 1 );
			stringsIndex.push_back( off );

			const uint32_t newId = (uint32_t)stringsHash.GetCount();
			stringsHash.SetAt( s, newId );
			return newId;
		}

		void addString( sTraceItem& rdi, const ItemName& name )
		{
			rdi.countFormatArgs = name.countArgs;
			rdi.stringIndex = addString( name.pointer );
			rdi.formatArgs = name.args;
		}

		std::vector<sTraceItem> items;
		uint64_t offset = 0;

	public:

		HRESULT create( LPCTSTR path )
		{
			CHECK( createDir( path ) );
			CHECK( file.Create( path, GENERIC_WRITE, 0, CREATE_ALWAYS ) );

			constexpr uint64_t cbHeader = sizeof( sFileHeader );
			CHECK( file.SetSize( cbHeader ) );
			CHECK( file.Seek( 0, SEEK_END ) );
			offset = 0;

			return S_OK;
		}

		HRESULT buffer( const ItemName& name, const void* rsi, size_t length, eDataType dt )
		{
			sTraceItem& rdi = items.emplace_back();
			const uint64_t cb = rdi.buffer( offset, length, dt );
			addString( rdi, name );
			assert( cb <= UINT_MAX );
			CHECK( file.Write( rsi, (DWORD)cb ) );
			offset += cb;
			return S_OK;
		}

		HRESULT tensor( const ItemName& name, const void* rsi, __m128i size, __m128i strides, eDataType dt )
		{
			sTraceItem& rdi = items.emplace_back();
			const uint64_t cb = rdi.tensor( offset, size, strides, dt );
			addString( rdi, name );
			assert( cb <= UINT_MAX );
			CHECK( file.Write( rsi, (DWORD)cb ) );
			offset += cb;
			return S_OK;
		}

		HRESULT close()
		{
			if( !file )
				return S_FALSE;

			const uint32_t cbStringsData = (uint32_t)stringsData.size();
			const uint32_t cbStringsIndex = (uint32_t)( stringsIndex.size() * 4 );
			if( !stringsIndex.empty() )
				CHECK( file.Write( stringsIndex.data(), cbStringsIndex ) );
			if( !stringsData.empty() )
				CHECK( file.Write( stringsData.data(), cbStringsData ) );

			const uint32_t cbItems = (uint32_t)items.size() * (uint32_t)sizeof( sTraceItem );
			if( !items.empty() )
				CHECK( file.Write( items.data(), cbItems ) );
			CHECK( file.Seek( 0, FILE_BEGIN ) );

			sFileHeader header;
			memset( &header, 0, sizeof( header ) );
			header.magic = header.correctMagic;
			header.cbItem = sizeof( sTraceItem );
			header.countItems = (uint32_t)items.size();
			header.bytesPayload = offset;
			header.countStrings = (uint32_t)stringsIndex.size();
			header.bytesStrings = cbStringsData + cbStringsIndex;
			CHECK( file.Write( &header, sizeof( header ) ) );
			CHECK( file.Flush() );
			file.Close();

			return S_OK;
		}
	};

	class TraceWriter : public iTraceWriter
	{
		TraceFileWriter file;

		HRESULT buffer( const ItemName& name, const void* rsi, size_t length, eDataType dt ) override final
		{
			return file.buffer( name, rsi, length, dt );
		}

		HRESULT tensor( const ItemName& name, const void* rsi, __m128i size, __m128i strides, eDataType dt ) override final
		{
			return file.tensor( name, rsi, size, strides, dt );
		}

	public:

		TraceWriter( LPCTSTR path )
		{
			check( file.create( path ) );
		}

		~TraceWriter()
		{
			check( file.close() );
		}
	};
}

std::unique_ptr<iTraceWriter> iTraceWriter::create( LPCTSTR path )
{
	return std::make_unique<TraceWriter>( path );
}

namespace
{
	static std::vector<float> tempFp32;
	static std::vector<uint16_t> tempFp16;

	template<class E>
	inline const void* ptr( const std::vector<E>& vec )
	{
		return vec.empty() ? nullptr : vec.data();
	}
}

HRESULT iTraceWriter::tensor( const ItemName& name, const DirectCompute::Tensor& source )
{
	const __m128i size = source.sizeVec();
	const __m128i strides = source.stridesVec();
	const eDataType dt = source.getType();
	if( dt == eDataType::FP32 )
	{
		source.download( tempFp32 );
		return tensor( name, ptr( tempFp32 ), size, strides, eDataType::FP32 );
	}
	else if( dt == eDataType::FP16 )
	{
		source.download( tempFp16 );
		return tensor( name, ptr( tempFp16 ), size, strides, eDataType::FP16 );
	}
	return E_NOTIMPL;
}

HRESULT iTraceWriter::tensor( const ItemName& name, const CpuCompute::Tensor& source )
{
	const __m128i size = source.sizeVec();
	const __m128i strides = source.stridesVec();
	const eDataType dt = source.type();

	if( dt == eDataType::FP32 )
		return tensor( name, source.fp32(), size, strides, eDataType::FP32 );
	else if( dt == eDataType::FP16 )
		return tensor( name, source.fp16(), size, strides, eDataType::FP16 );
	else
		return E_NOTIMPL;
}

#if BUILD_BOTH_VERSIONS
#include "../../source/ggml.h"
HRESULT __declspec( noinline ) iTraceWriter::tensor( const ItemName& name, const ggml_tensor& source )
{
	__m128i size = load16( source.ne );
	__m128i strides = _mm_setr_epi32(
		(int)(uint32_t)source.nb[ 0 ],
		(int)(uint32_t)source.nb[ 1 ],
		(int)(uint32_t)source.nb[ 2 ],
		(int)(uint32_t)source.nb[ 3 ] );

	const __m128i ones = _mm_set1_epi32( 1 );
	switch( source.n_dims )
	{
	case 0:
		size = ones;
		break;
	case 1:
		size = _mm_blend_epi16( size, ones, 0b11111100 );
		break;
	case 2:
		size = _mm_blend_epi16( size, ones, 0b11110000 );
		break;
	case 3:
		size = _mm_blend_epi16( size, ones, 0b11000000 );
		break;
	case 4:
		break;
	default:
		return E_INVALIDARG;
	}

	const ggml_type dt = source.type;
	switch( dt )
	{
	case GGML_TYPE_F16:
		strides = _mm_srli_epi32( strides, 1 );
		return tensor( name, source.data, size, strides, eDataType::FP16 );
	case GGML_TYPE_F32:
		strides = _mm_srli_epi32( strides, 2 );
		return tensor( name, source.data, size, strides, eDataType::FP32 );
	default:
		return E_NOTIMPL;
}
}
#else
HRESULT iTraceWriter::tensor( const ItemName& name, const ggml_tensor& source )
{
	return E_NOTIMPL;
}
#endif