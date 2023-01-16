#pragma once
#include "../D3D/device.h"
#include "TensorShape.h"

namespace DirectCompute
{
	// 96 bytes dynamic constant buffers, with dimensions and VRAM layout of 2-3 tensors
	class ConstantBuffer
	{
		CComPtr<ID3D11Buffer> buffer;

	public:
		HRESULT create();
		HRESULT update( const TensorShape& t0 );
		HRESULT update( const TensorShape& t0, const TensorShape& t1 );
		HRESULT update( const TensorShape& t0, const TensorShape& t1, const TensorShape& t2 );

		void bind() const;

		__m128i getMemoryUse() const
		{
			return bufferMemoryUsage( buffer );
		}
	};
}