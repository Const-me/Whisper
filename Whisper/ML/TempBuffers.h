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

			HRESULT resize( DXGI_FORMAT format, size_t elements, size_t cbElement, bool zeroMemory, CComPtr<ID3D11Buffer>& cb );

			size_t getCapacity() const { return capacity; }
		};

		Buffer m_fp16;
		Buffer m_fp16_2;
		Buffer m_fp32;

	public:

		CComPtr<ID3D11Buffer> smallCb;

		static void zeroMemory( ID3D11UnorderedAccessView* uav, uint32_t length, CComPtr<ID3D11Buffer>& cb );

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