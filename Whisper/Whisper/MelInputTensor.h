#pragma once
#include "../ML/TensorEx.h"
#include "sEncodeParams.h"
#include "iSpectrogram.h"

namespace DirectCompute
{
	// Input tensor in VRAM, in a dynamic FP32 buffer
	class MelInputTensor : public TensorEx
	{
		uint32_t capacity;

	public:

		HRESULT create( Whisper::iSpectrogram& spectrogram, const sEncodeParams& encParams );

		__m128i getMemoryUse() const
		{
			return setHigh_size( (size_t)capacity * 4 );
		}
	};
}