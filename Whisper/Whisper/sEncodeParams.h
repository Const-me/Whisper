#pragma once
#include <stdint.h>

namespace DirectCompute
{
	struct sEncodeParams
	{
		uint32_t n_ctx, n_mels, mel_offset;
		uint32_t layersCount, n_state, n_head;
		uint32_t n_audio_ctx, n_text_state, n_text_layer, n_text_ctx;
	};

	struct sDecodeParams
	{
		uint32_t n_state, n_head;
		uint32_t n_ctx, n_past, M;
		uint32_t n_text_layer;
		uint32_t n_vocab;
	};
}