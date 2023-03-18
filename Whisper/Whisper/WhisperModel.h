#pragma once
#include <memory>
#include "Vocabulary.h"
#include "ModelBuffers.h"
#include "../../ComLightLib/streams.h"
#include "../CPU/DecoderTensors.h"
#include "../API/sLoadModelCallbacks.h"
#include "sModelParams.h"

namespace Whisper
{
	struct Filters
	{
		uint32_t n_mel;
		uint32_t n_fft;
		std::vector<float> data;
	};

	struct ModelShared
	{
		Vocabulary vocab;
		Filters filters;
#if BUILD_HYBRID_VERSION
		CpuCompute::DecoderTensors hybridTensors;
#endif
	};

	// The complete model, as loaded from a GGML binary file.
	// The entire model is immutable, and can be safely used from multiple threads in parallel.
	// The tensors are uploaded to VRAM and don’t stay in system memory, everything else is in the system RAM.
	struct WhisperModel
	{
		sModelParams parameters;
		std::shared_ptr<ModelShared> shared;
		DirectCompute::ModelBuffers tensors;

		HRESULT load( ComLight::iReadStream* stm, bool hybrid, const sLoadModelCallbacks* callbacks );
		HRESULT createClone( const WhisperModel& rsi );

		// A vector of 2 uint64_t values, both numbers are 100 nanosecond ticks:
		// 0. The time it took to load the model, measured on CPU
		// 1. The time it took to upload all these tensors to VRAM, measured on GPU
		__m128i getLoadTimes() const
		{
			static_assert( offsetof( WhisperModel, loadTimeCpu ) + 8 == offsetof( WhisperModel, loadTimeGpu ) );
			return _mm_loadu_si128( ( const __m128i* )( &loadTimeCpu ) );
		}

		__m128i getMemoryUse() const;

	private:
		uint64_t loadTimeCpu = 0;
		uint64_t loadTimeGpu = 0;

		class CallbacksImpl;

		HRESULT loadGpu( ComLight::iReadStream* stm, CallbacksImpl& callbacks );
		HRESULT loadHybrid( ComLight::iReadStream* stm, CallbacksImpl& callbacks );
	};
}