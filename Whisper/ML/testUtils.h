#pragma once
#include "../D3D/downloadBuffer.h"
#include "../D3D/RenderDoc/renderDoc.h"
#include <unordered_set>
#include <functional>

// Funfact: this code written by ChatGPT
namespace std
{
	template<>
	struct hash<array<uint32_t, 8>>
	{
		size_t operator()( const array<uint32_t, 8>& arr ) const
		{
			size_t result = 0;
			for( uint32_t element : arr )
				result = ( result * 31 ) ^ element;
			return result;
		}
	};
}

namespace DirectCompute
{
	struct sTensorDiff
	{
		// maximum( absolute( a - b ) )
		float maxAbsDiff;
		// average( ( a - b )^2 )
		float avgDiffSquared;
		size_t length;

		void print() const;
		void print( const char* what ) const;
	};

	// Compute difference between 2 FP32 vectors
	sTensorDiff computeDiff( const float* a, const float* b, size_t length );

	// Compute difference between 2 FP16 vectors
	sTensorDiff computeDiff( const uint16_t* a, const uint16_t* b, size_t length );

	class Tensor;
	sTensorDiff computeDiff( const Tensor& a, const Tensor& b );

	HRESULT dbgWriteBinaryFile( LPCTSTR fileName, const void* rsi, size_t cb );

	// Print unique sizes of the two tensors
	class PrintUniqueTensorSizes
	{
		std::unordered_set<std::array<uint32_t, 8>> set;
		const char* const what;
		void printImpl( const std::array<uint32_t, 8>& a );

	public:
		PrintUniqueTensorSizes( const char* w ) : what( w ) { }

		void print( const Tensor& lhs, const Tensor& rhs );
		void print( const Tensor& lhs );
		void print( const int* lhs, const int* rhs );
	};
}