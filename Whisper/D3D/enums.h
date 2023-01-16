#pragma once
#include <stdint.h>
#include <assert.h>

namespace DirectCompute
{
	enum struct eDataType : uint8_t
	{
		FP16,
		FP32,
		U32,
	};

	inline size_t elementSize( eDataType dt )
	{
		assert( dt == eDataType::FP16 || dt == eDataType::FP32 || dt == eDataType::U32 );

		return ( dt == eDataType::FP16 ) ? 2 : 4;
	}

	DXGI_FORMAT viewFormat( eDataType dt );

	enum struct eBufferUse : uint8_t
	{
		// Immutable tensor, readable from GPU
		Immutable,
		// Read+write tensor, readable and writable on GPU
		ReadWrite,
		// Read+write tensor, readable and writable on GPU, which supports downloads from GPU
		ReadWriteDownload,
		// The tensor is accessible by both GPU (read only) and CPU (write only). Optimized for resources frequently updated from CPU.
		Dynamic,
	};
}