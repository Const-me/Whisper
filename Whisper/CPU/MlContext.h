#pragma once
#include "Tensor.h"
#include "ParallelForRunner.h"

namespace CpuCompute
{
	class MlContext
	{
		ParallelForRunner pfor;
		iMemoryAllocator* allocator = nullptr;

	public:
		MlContext( int threads );
		MlContext( const MlContext& ) = delete;
		~MlContext() = default;

		HRESULT setThreadsCount( int threads )
		{
			return pfor.setThreadsCount( threads );
		}

		iMemoryAllocator* setAllocator( iMemoryAllocator* alloc )
		{
			iMemoryAllocator* const ret = allocator;
			allocator = alloc;
			return ret;
		}

		Tensor createTensor( eDataType type, const std::array<uint32_t, 4>& size );
		Tensor createTensor( eDataType type, std::initializer_list<uint32_t> size );

		Tensor addRows( const Tensor& d_te, const Tensor& d_pe, const int* tokens, const int n_tokens, const int n_past );

		Tensor norm( const Tensor& arg );

		// cur = add( mul( repeat( w, cur ), cur ), repeat( b, cur ) );
		void fmaRepeat( Tensor& cur, const Tensor& w, const Tensor& b );

		inline void fmaRepeat( Tensor& cur, const TensorPair wb )
		{
			fmaRepeat( cur, wb.w, wb.b );
		}

		// Multiply two matrices
		Tensor mulMat( const Tensor& a, const Tensor& b );

		// cur = add( repeat( b, cur ), cur ); cur = scale(cur, scaling)
		void addRepeatScale( Tensor& cur, const Tensor& b, float scaling );

		void addRepeat( Tensor& cur, const Tensor& b );

		Tensor add( const Tensor& a, const Tensor& b );
		void addInPlace( Tensor& a, const Tensor& b );
		void addRepeatGelu( Tensor& cur, const Tensor& b );

		// cur = scale(cur, scaling)
		void scale( Tensor& cur, float scaling );

		void diagMaskInf( Tensor& cur, uint32_t n_past );

		void softMax( Tensor& cur, float inputScale = 1.0f );

		Tensor copy( const Tensor& a, eDataType type, std::initializer_list<uint32_t> size );

		HRESULT copyImpl( Tensor& result, const Tensor& source );

		Tensor permute( const Tensor& a, uint8_t axis0, uint8_t axis1, uint8_t axis2, uint8_t axis3 );

		void copyInPlace( Tensor& dest, const Tensor& a, eDataType type, std::initializer_list<uint32_t> size );
	};
}