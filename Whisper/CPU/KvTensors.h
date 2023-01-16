#pragma once
#include "Tensor.h"
#include "LargeBuffer.h"
#include "../Whisper/sModelParams.h"

namespace CpuCompute
{
	class KvTensors
	{
		uint16_t* keys = nullptr;
		uint16_t* values = nullptr;
		uint32_t size = 0;

		CpuCompute::LargeBuffer memory;

	public:
		// Create these two large tensors, FP16 precision
		HRESULT create( const Whisper::sModelParams& mp );

		// A slice of model.memory_cross_k tensor
		Tensor keysView( uint32_t len, uint32_t off ) const
		{
			if( len + off <= size )
				return Tensor::fromData( keys + off, eDataType::FP16, len );
			throw E_BOUNDS;
		}

		// A slice of model.memory_cross_v tensor
		Tensor valuesView( uint32_t len, uint32_t off ) const
		{
			if( len + off <= size )
				return Tensor::fromData( values + off, eDataType::FP16, len );
			throw E_BOUNDS;
		}
	};
}