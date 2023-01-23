#pragma once
#include <stdint.h>
#include <array>
#include <smmintrin.h>

struct ggml_tensor;
using HRESULT = long;

namespace DirectCompute
{
	// This POD structure describes the shape of a tensor.
	// It’s used for both GPU tensors in VRAM, and tensors in system memory used by the Hybrid model.
	struct TensorShape
	{
		// Count of elements, up to 4 coordinates
		// The unused coordinates are set to 1
		std::array<uint32_t, 4> ne;

		// Strides of the tensor
		// For a dense row-major tensor, these numbers are [ 1, ne[0], ne[0]*ne[1], ne[0]*ne[1]*ne[2] ]
		// Note that unlike GGML code, these numbers are expressed in elements not bytes, but the meaning is the same
		// GPU matrices reshaped into panels are keeping different values here: [ 0, panelSize, panelSize * panelsCount, panelSize * panelsCount * ne[ 2 ] ]
		std::array<uint32_t, 4> nb;

		TensorShape();
		TensorShape( const TensorShape& that );
		void operator=( const TensorShape& that );
		HRESULT create( const ggml_tensor& ggml );
		TensorShape( const ggml_tensor& ggml );

		__m128i __vectorcall sizeVec() const
		{
			return load( ne );
		}
		__m128i __vectorcall stridesVec() const
		{
			return load( nb );
		}

		uint32_t countRows() const
		{
			return ne[ 1 ] * ne[ 2 ] * ne[ 3 ];
		}

		uint32_t countElements() const
		{
			// return ne[ 0 ] * countRows();
			const __m128i a = sizeVec();
			const __m128i b = _mm_srli_si128( a, 4 );
			const __m128i p2 = _mm_mul_epu32( a, b );
			uint64_t res = (uint64_t)_mm_extract_epi64( p2, 1 );
			res *= (uint64_t)_mm_cvtsi128_si64( p2 );
			assert( 0 == ( res >> 32 ) );
			return (uint32_t)res;
		}

		// Compute strides from sizes, assuming dense row-major memory layout of the tensor
		void setDenseStrides();

		bool isMatrix() const
		{
			// return ne[ 2 ] == 1 && ne[ 3 ] == 1;
			const uint64_t num = *(const uint64_t*)&ne[ 2 ];
			return num == 0x100000001ull;
		}
		bool isVector() const
		{
			return 1 == ne[ 1 ] && isMatrix();
		}

		// True of this tensor is dense and row-major
		bool isContinuous() const
		{
			/* return 1 == nb[ 0 ] &&
				nb[ 1 ] == nb[ 0 ] * ne[ 0 ] &&
				nb[ 2 ] == nb[ 1 ] * ne[ 1 ] &&
				nb[ 3 ] == nb[ 2 ] * ne[ 2 ]; */

			const __m128i nbv = stridesVec();
			const __m128i nev = sizeVec();
			__m128i tmp = _mm_mullo_epi32( nbv, nev );	// Vertical product of int32 lanes
			tmp = _mm_shuffle_epi32( tmp, _MM_SHUFFLE( 2, 1, 0, 0 ) );	// Shift left by 1 int32 lane
			tmp = _mm_insert_epi32( tmp, 1, 0 );	// Reset X lane to 1
			return vectorEqual( tmp, nbv );
		}

		// Reset all fields to zero
		void setZero()
		{
			const __m128i z = _mm_setzero_si128();
			_mm_storeu_si128( ( __m128i* )ne.data(), z );
			_mm_storeu_si128( ( __m128i* )nb.data(), z );
		}
	};

	// True when two tensors have equal count of elements
	inline bool isSameShape( const TensorShape& t0, const TensorShape& t1 )
	{
		__m128i a = t0.sizeVec();
		__m128i b = t1.sizeVec();
		return vectorEqual( a, b );
	}

	// True when two tensors have equal count of elements, and equal VRAM layout too
	inline bool isSameShapeAndLayout( const TensorShape& t0, const TensorShape& t1 )
	{
		__m128i a, b, x;
		a = t0.sizeVec();
		b = t1.sizeVec();
		x = _mm_xor_si128( a, b );

		a = t0.stridesVec();
		b = t1.stridesVec();
		x = _mm_or_si128( x, _mm_xor_si128( a, b ) );
		return (bool)_mm_testz_si128( x, x );
	}

	// True when we can multiply two tensors of the provided shapes
	bool canMulMat( const TensorShape& t0, const TensorShape& t1 );
}