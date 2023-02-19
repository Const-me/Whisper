#pragma once
#include "../ML/MlContext.h"
#include "MelInputTensor.h"
#include "KeyValueBuffers.h"
#include "sEncodeParams.h"
#include "DecoderInputBuffers.h"
#include "DecoderResultBuffer.h"
#include "../ML/TensorsArena.h"
#include "iSpectrogram.h"
#include "../Hybrid/HybridContext.h"
#include <memory>
#include "WhisperModel.h"
#include <tuple>
#include <optional>

namespace DirectCompute
{
	struct TensorPair;
	struct ModelBuffers;

	class WhisperContext : public MlContext
	{
		struct Arenas
		{
			TensorsArena outer;
			TensorsArena layer;
			Arenas();
			__m128i getMemoryUse() const;
		};

		iTensorArena* currentArena = nullptr;
		Arenas arenas;

		// Specialized tensor arena for decoder layer outputs, with just a single tensor
		class DecoderLayerPool : public iTensorArena
		{
			PooledTensor result;
		public:
			Tensor tensor( eDataType type, const std::array<uint32_t, 4>& ne ) override final;
			void reset() override final { }
			__m128i getMemoryUse() const;
			void clear()
			{
				result.clear();
			}
			HRESULT zeroMemory()
			{
				return result.zeroMemory();
			}
		};

		DecoderLayerPool decPool;

		class ArenaRaii;

		MelInputTensor melInput;
		KeyValueBuffers kv, kvCross;
		DecoderInputBuffers decoderInput;
		DecoderResultBuffer decoderOutput;
		const ModelBuffers& gpuModel;
#if BUILD_HYBRID_VERSION
		std::unique_ptr<HybridContext> hybridContext;
#endif
		struct sWhisperMel
		{
			uint32_t n_len;
			uint32_t n_mel;
			const std::vector<float>& data;
		};

		void createKeyValueBuffers( const sEncodeParams& encParams );
		// Encoder methods
		Tensor convolutionAndGelu( const Tensor& mel, uint32_t n_ctx );
		Tensor encodeLayer( const Tensor& source, size_t index, uint32_t n_state, uint32_t n_head, uint32_t n_ctx );

		struct sLayerDecParams;

		// Decoder methods
		Tensor decodeLayer( const Tensor& source, size_t index, const sLayerDecParams& ldp );

		// cur = add( mul( repeat( that.w, cur ), cur ), repeat( that.b, cur ) );
		void fmaRepeat( Tensor& cur, const TensorPair& that );

		Tensor createTensor( eDataType type, const std::array<uint32_t, 4>& ne ) override final;

	public:
#if BUILD_BOTH_VERSIONS
		WhisperContext();
		~WhisperContext();
#else
		~WhisperContext() = default;
#endif
		WhisperContext( const Whisper::WhisperModel& wm, Whisper::ProfileCollection& pc );
		WhisperContext( const WhisperContext& ) = delete;

		Tensor encode( Whisper::iSpectrogram& spectrogram, const sEncodeParams& encParams );

		void decode( const int* tokens, const int n_tokens, const sDecodeParams& decParams, std::vector<float>& probs, int threads );

		static WhisperContext& current();

		// Create a RAII object which measures both CPU and GPU time for the complete runFull() method
		decltype( auto ) completeProfiler()
		{
			return std::make_tuple(
				profiler.cpuBlock( Whisper::eCpuBlock::Run ),
				profiler.block( eProfilerBlock::Run ) );
		}

		// Create a RAII object which measures CPU and optionally GPU time for the loop which calls decode() method
		decltype( auto ) decodeProfiler()
		{
#if BUILD_HYBRID_VERSION
			if( hybridContext )
				return std::make_tuple(
					profiler.cpuBlock( Whisper::eCpuBlock::Decode ),
					std::optional<GpuProfiler::BlockRaii>{} );
			else
				return std::make_tuple(
					profiler.cpuBlock( Whisper::eCpuBlock::Decode ),
					std::optional<GpuProfiler::BlockRaii>{ std::in_place, profiler.block( eProfilerBlock::Decode ) } );
#else
			return std::make_tuple(
				profiler.cpuBlock( Whisper::eCpuBlock::Decode ),
				profiler.block( eProfilerBlock::Decode ) );
#endif
		}

		__m128i getMemoryUse() const;

		HRESULT clearState();
	};
}