#pragma once
#include "Tensor.h"

namespace DirectCompute
{
	// A tensor which supports dynamic updates from CPU, or downloads from VRAM to system RAM
	class TensorEx : public Tensor
	{
	protected:
		CComPtr<ID3D11Buffer> buffer;
		CComPtr<ID3D11Buffer> stagingBuffer;

		HRESULT getViewSize( uint32_t& cbElement, uint32_t& countElements ) const;

	public:

		HRESULT create( const ggml_tensor& ggml, eBufferUse usage, bool uploadData );
		HRESULT create( eDataType type, eBufferUse usage, const std::array<uint32_t, 4>& sizeElements );

		HRESULT download( void* rdi, size_t cb ) const;

		HRESULT download( void* rdi ) const;

		template<class E>
		HRESULT download( std::vector<E>& vec ) const
		{
			uint32_t cbElement, numElements;
			CHECK( getViewSize( cbElement, numElements ) );

			try
			{
				vec.resize( numElements );
			}
			catch( const std::bad_alloc& )
			{
				return E_OUTOFMEMORY;
			}

			return download( vec.data(), (size_t)cbElement * numElements );
		}
	};
}