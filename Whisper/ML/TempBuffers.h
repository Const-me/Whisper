#pragma once
#include "TensorGpuViews.h"

namespace DirectCompute
{
	class TempBuffers
	{
		class Buffer : public TensorGpuViews
		{
			size_t capacity = 0;

		public:

			void clear()
			{
				TensorGpuViews::clear();
				capacity = 0;
			}

			HRESULT resize( DXGI_FORMAT format, size_t elements, size_t cbElement, bool zeroMemory );

			size_t getCapacity() const { return capacity; }
		};

		Buffer m_fp16;
		Buffer m_fp16_2;
		Buffer m_fp32;

	public:

		const TensorGpuViews& fp16( size_t countElements, bool zeroMemory = false );
		const TensorGpuViews& fp16_2( size_t countElements, bool zeroMemory = false );
		const TensorGpuViews& fp32( size_t countElements, bool zeroMemory = false );

		void clear()
		{
			m_fp16.clear();
			m_fp16_2.clear();
			m_fp32.clear();
		}

		__m128i getMemoryUse() const;
	};
}