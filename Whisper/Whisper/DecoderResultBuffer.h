#pragma once
#include "../ML/Tensor.h"

namespace DirectCompute
{
	class DecoderResultBuffer
	{
		CComPtr<ID3D11Buffer> buffer;
		uint32_t m_size = 0;
		uint32_t m_capacity = 0;

	public:

		void copyFromVram( const Tensor& rsi );

		void copyToVector( std::vector<float>& vec ) const;

		uint32_t size() const
		{
			return m_size;
		}
		void clear();

		__m128i getMemoryUse() const
		{
			return bufferMemoryUsage( buffer );
		}
	};
}