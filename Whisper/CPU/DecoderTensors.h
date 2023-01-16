#pragma once
#include <vector>
#include "Tensor.h"
#include "LargeBuffer.h"
#if TENSOR_GGML_COMPAT
#include "../source/ggml.h"
#endif

namespace CpuCompute
{
	// A set of tensors for one decoder's layer
	struct LayerDecoder
	{
		// decoder.blocks.*.attn_ln
		TensorPair attnLn0;
		// decoder.blocks.*.attn.out
		TensorPair attnLn1;
		// decoder.blocks.*.attn.query
		TensorPair attnQuery;
		// decoder.blocks.*.attn.key
		Tensor attnKey;
		// decoder.blocks.*.attn.value
		TensorPair attnValue;
		// decoder.blocks.*.cross_attn_ln
		TensorPair crossAttnLn0;
		// decoder.blocks.*.cross_attn.out
		TensorPair crossAttnLn1;
		// decoder.blocks.*.cross_attn.query
		TensorPair crossAttnQuery;

		// decoder.blocks.*.cross_attn.key
		// Tensor crossAttnKey;
		// decoder.blocks.*.cross_attn.value
		// TensorPair crossAttnValue;

		// decoder.blocks.*.mlp_ln
		TensorPair mlpLn;
		// decoder.blocks.*.mlp.0
		TensorPair mlp0;
		// decoder.blocks.*.mlp.2
		TensorPair mlp1;

#if TENSOR_GGML_COMPAT
		// decoder.blocks.*.attn_ln
		ggml_tensor* attn_ln_0_w;
		ggml_tensor* attn_ln_0_b;

		// decoder.blocks.*.attn.out
		ggml_tensor* attn_ln_1_w;
		ggml_tensor* attn_ln_1_b;

		// decoder.blocks.*.attn.query
		ggml_tensor* attn_q_w;
		ggml_tensor* attn_q_b;

		// decoder.blocks.*.attn.key
		ggml_tensor* attn_k_w;

		// decoder.blocks.*.attn.value
		ggml_tensor* attn_v_w;
		ggml_tensor* attn_v_b;

		// decoder.blocks.*.cross_attn_ln
		ggml_tensor* cross_attn_ln_0_w;
		ggml_tensor* cross_attn_ln_0_b;

		// decoder.blocks.*.cross_attn.out
		ggml_tensor* cross_attn_ln_1_w;
		ggml_tensor* cross_attn_ln_1_b;

		// decoder.blocks.*.cross_attn.query
		ggml_tensor* cross_attn_q_w;
		ggml_tensor* cross_attn_q_b;

		// decoder.blocks.*.mlp_ln
		ggml_tensor* mlp_ln_w;
		ggml_tensor* mlp_ln_b;

		// decoder.blocks.*.mlp.0
		ggml_tensor* mlp_0_w;
		ggml_tensor* mlp_0_b;

		// decoder.blocks.*.mlp.2
		ggml_tensor* mlp_1_w;
		ggml_tensor* mlp_1_b;
#endif
	};

	struct DecoderTensors
	{
		// decoder.positional_embedding
		Tensor positionalEmbedding;

		// decoder.token_embedding
		Tensor tokenEmbedding;

		// decoder.ln
		TensorPair ln;
		// A vector of layers
		std::vector<LayerDecoder> layers;

		void setMemoryBuffer( LargeBuffer&& mem ) noexcept
		{
			memory = std::move( mem );
#if TENSOR_GGML_COMPAT
			makeCompatTensors();
#endif
		}

#if TENSOR_GGML_COMPAT
		void makeCompatTensors();

		// decoder.positional_embedding
		ggml_tensor* d_pe; // DD

		// decoder.token_embedding
		ggml_tensor* d_te; // DD

		// decoder.ln
		ggml_tensor* d_ln_w; // DD
		ggml_tensor* d_ln_b; // DD
#endif

	private:
		// A smart pointer which owns the memory for all the above tensors
		LargeBuffer memory;
#if TENSOR_GGML_COMPAT
		std::vector<ggml_tensor> ggml;
#endif
	};
}