#pragma once
#include <memory>
#include "../../D3D/enums.h"

namespace DirectCompute
{
	class Tensor;
}
namespace CpuCompute
{
	class Tensor;
}

struct ggml_tensor;

namespace Tracing
{
	using DirectCompute::eDataType;

	struct ItemName
	{
		const char* pointer;
		std::array<uint32_t, 4> args;
		uint8_t countArgs;

		ItemName( const char* str )
		{
			pointer = str;
			_mm_storeu_si128( ( __m128i* )args.data(), _mm_setzero_si128() );
			countArgs = 0;
		}
		ItemName( const char* str, int a0 )
		{
			pointer = str;
			__m128i v = _mm_cvtsi32_si128( a0 );
			_mm_storeu_si128( ( __m128i* )args.data(), v );
			countArgs = 1;
		}
		ItemName( const char* str, uint32_t a0 )
		{
			pointer = str;
			__m128i v = _mm_cvtsi32_si128( (int)a0 );
			_mm_storeu_si128( ( __m128i* )args.data(), v );
			countArgs = 1;
		}
		ItemName( const char* str, size_t a0 )
		{
			pointer = str;
			__m128i v = _mm_cvtsi32_si128( (int)a0 );
			_mm_storeu_si128( ( __m128i* )args.data(), v );
			countArgs = 1;
		}
	};

	class iTraceWriter
	{
	public:
		virtual ~iTraceWriter() {}

		static std::unique_ptr<iTraceWriter> create( LPCTSTR path );

		virtual HRESULT buffer( const ItemName& name, const void* rsi, size_t length, eDataType dt ) = 0;

		virtual HRESULT tensor( const ItemName& name, const void* rsi, __m128i size, __m128i strides, eDataType dt ) = 0;

		HRESULT tensor( const ItemName& name, const DirectCompute::Tensor& tensor );
		HRESULT tensor( const ItemName& name, const CpuCompute::Tensor& tensor );
		HRESULT tensor( const ItemName& name, const ggml_tensor& tensor );
	};
}