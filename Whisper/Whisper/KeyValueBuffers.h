#pragma once
#include "../ML/Tensor.h"

namespace DirectCompute
{
	// FP16 buffer for self-attention and cross-attention layers
	class AttentionBuffer
	{
		CComPtr<ID3D11Buffer> buffer;
		uint32_t m_size = 0;

	public:
		// Create buffer for the specified count of elements
		void resize( uint32_t size );

		// Create an 1D tensor which references a slice of that buffer
		Tensor view( uint32_t length, uint32_t offset ) const;

		void clear()
		{
			buffer = nullptr;
			m_size = 0;
		}

		ID3D11Buffer* getBuffer() const { return buffer; }

		uint32_t getSize() const { return m_size; }

		HRESULT zeroMemory() const;
	};

	struct KeyValueBuffers
	{
		AttentionBuffer keys, values;

		void resize( uint32_t size );

		void clear()
		{
			keys.clear();
			values.clear();
		}

		__m128i getMemoryUse() const
		{
			size_t i = keys.getSize();
			i += values.getSize();
			i *= sizeof( uint16_t );
			return setHigh_size( (int64_t)i );	// They both are in VRAM
		}

		HRESULT zeroMemory() const;
	};
}