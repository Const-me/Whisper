#pragma once
#include "../ML/Tensor.h"

namespace DirectCompute
{
	// A dynamic buffer
	class DecoderInputBuffers
	{
		CComPtr<ID3D11Buffer> embd;
		uint32_t m_size = 0;
		uint32_t m_capacity = 0;

	public:

		void resize( uint32_t size );

		// Create 1D tensor with R32_UINT elements, upload the source data
		Tensor embedding( const int* rsi ) const;

		void clear();

		__m128i getMemoryUse() const
		{
			size_t i = m_capacity;
			i *= sizeof( uint32_t );
			return _mm_set_epi64x( (int64_t)i, 0 );
		}

		HRESULT zeroMemory() const;
	};
}