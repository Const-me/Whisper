#pragma once
#include "../ML/Tensor.h"
#include <vector>

namespace DirectCompute
{
	// A pair of tensors containing weights and biases; apparently, both tensors are of the same shape.
	struct TensorPair
	{
		Tensor w, b;

		__m128i getMemoryUse() const;
	};

	// A set of tensors for one encoder's layer
	struct LayerEncoder
	{
		// encoder.blocks.*.attn_ln
		TensorPair attnLn0;
		// encoder.blocks.*.attn.out
		TensorPair attnLn1;
		// encoder.blocks.*.attn.query
		TensorPair attnQuery;
		// encoder.blocks.*.attn.key
		Tensor attnKey;
		// encoder.blocks.*.attn.value
		TensorPair attnValue;
		// encoder.blocks.*.mlp_ln
		TensorPair mlpLn;
		// encoder.blocks.*.mlp.0
		TensorPair mlp0;
		// encoder.blocks.*.mlp.2
		TensorPair mlp1;

		__m128i getMemoryUse() const;
	};

	// A set of tensors for the encoder
	struct EncoderBuffers
	{
		// encoder.positional_embedding
		Tensor positionalEmbedding;
		// encoder.conv1
		TensorPair conv1;
		// encoder.conv2
		TensorPair conv2;
		// encoder.ln_post
		TensorPair lnPost;
		// A vector of layers
		std::vector<LayerEncoder> layers;

		__m128i getMemoryUse() const;
	};

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
		Tensor crossAttnKey;
		// decoder.blocks.*.cross_attn.value
		TensorPair crossAttnValue;
		// decoder.blocks.*.mlp_ln
		TensorPair mlpLn;
		// decoder.blocks.*.mlp.0
		TensorPair mlp0;
		// decoder.blocks.*.mlp.2
		TensorPair mlp1;

		__m128i getMemoryUse() const;
	};

	// A set of tensors for the decoder
	struct DecoderBuffers
	{
		// decoder.positional_embedding
		Tensor positionalEmbedding;
		// decoder.token_embedding
		Tensor tokenEmbedding;
		// decoder.ln
		TensorPair ln;
		// A vector of layers
		std::vector<LayerDecoder> layers;

		__m128i getMemoryUse() const;
	};

	// A complete set of tensors for a model
	struct ModelBuffers
	{
		EncoderBuffers enc;
		DecoderBuffers dec;
		__m128i getMemoryUse() const;

		HRESULT createClone( const ModelBuffers& rsi );
	};

#if BUILD_BOTH_VERSIONS
	extern const ModelBuffers& gpuModel;
#endif
}