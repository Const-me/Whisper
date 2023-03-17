#pragma once
#include "TensorShape.h"
#include "TensorGpuViews.h"
#include "../D3D/enums.h"

namespace DirectCompute
{
	// A minimal tensor object sufficient to compute things on GPU, with compute shaders
	// This class only takes 48 bytes in system memory, and is very cheap to make copies 'coz GPU objects are reference counted.
	class Tensor : public TensorShape, public TensorGpuViews
	{
		CComPtr<ID3D11Buffer> getBuffer() const;

		struct TensorType
		{
			eDataType type;
			eBufferUse usage;
			bool hasInitialData;
		};
#ifdef _DEBUG
		// In debug builds, we include a few pieces of data to this class.
		TensorType dbgType;
#endif
	protected:
		HRESULT create( eDataType type, std::initializer_list<uint32_t> sizeElements, eBufferUse usage, CComPtr<ID3D11Buffer>& buffer, const void* rsi, ID3D11Buffer** ppStagingBuffer, bool shared = false );

		static uint32_t dxgiSizeof( DXGI_FORMAT format );

		void downloadImpl( const D3D11_SHADER_RESOURCE_VIEW_DESC& viewDesc, uint32_t countElements, size_t cbElement, void* rdi ) const;

	public:
		Tensor() = default;

		// These copy operators don't copy any data, they merely increment ref.counter of the GPU resources
		Tensor( const Tensor& );
		Tensor( Tensor&& that ) noexcept;
		Tensor& operator=( const Tensor& that );
		Tensor& operator=( Tensor&& that ) noexcept;

		// Move the provided buffer views into this newly created tensor, and assign the shape
		// This destroys old values in the smart pointers
		Tensor( const TensorShape& shape, CComPtr<ID3D11ShaderResourceView>& srv, CComPtr<ID3D11UnorderedAccessView>& uav ) noexcept;

		Tensor( const TensorShape& shape, const TensorGpuViews& views );

		// Create a tensor from the GGML's one
		HRESULT create( const ggml_tensor& ggml, eBufferUse usage, bool uploadData );

		// Create a new dense tensor of the specified size in elements, without initial data
		HRESULT create( eDataType type, std::initializer_list<uint32_t> sizeElements, bool shared = false );
		HRESULT create( eDataType type, const std::array<uint32_t, 4>& sizeElements, bool shared = false );
		HRESULT createImmutable( eDataType type, const std::array<int, 4>& size, const void* rsi );

		eDataType getType() const;

		// This method should probably only be used to test things
		// TensorEx is better for production usage, because it creates staging buffer in advance.
		void download( std::vector<float>& vec ) const;
		void download( std::vector<uint16_t>& vec ) const;

		// ggml_reshape_3d
		Tensor reshape3d( uint32_t ne0, uint32_t ne1, uint32_t ne2 ) const;

		inline void dbgSetType( eDataType dt, bool hasData = false, eBufferUse use = eBufferUse::ReadWrite )
		{
#ifdef _DEBUG
			dbgType.type = dt;
			dbgType.hasInitialData = hasData;
			dbgType.usage = use;
#endif
		}

		__m128i getMemoryUse() const
		{
			return resourceMemoryUsage( srv );
		}
	};
}