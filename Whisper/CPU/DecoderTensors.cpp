#include "stdafx.h"
#include "DecoderTensors.h"
using namespace CpuCompute;

#if TENSOR_GGML_COMPAT
namespace
{
	class CompatContext
	{
		std::vector<ggml_tensor>& vec;
		size_t index;

	public:
		CompatContext( std::vector<ggml_tensor>& dest, size_t layers ) :
			vec( dest )
		{
			constexpr size_t tensorsPerLayer = 21;
			const size_t count = tensorsPerLayer * layers + 4;
			vec.resize( count );
			index = 0;
		}

		void add( const Tensor& rsi, ggml_tensor*& res )
		{
			ggml_tensor& ten = vec[ index ];
			index++;
			ten = rsi.ggml();
			res = &ten;
		}

		void add2( const TensorPair& rsi, ggml_tensor*& w, ggml_tensor*& b )
		{
			add( rsi.w, w );
			add( rsi.b, b );
		}

		bool isComplete() const
		{
			return index == vec.size();
		}
	};
}

void DecoderTensors::makeCompatTensors()
{
	CompatContext ctx( ggml, layers.size() );

	ctx.add( positionalEmbedding, d_pe );
	ctx.add( tokenEmbedding, d_te );
	ctx.add2( ln, d_ln_w, d_ln_b );

	for( auto& i : layers )
	{
		ctx.add2( i.attnLn0, i.attn_ln_0_w, i.attn_ln_0_b );
		ctx.add2( i.attnLn1, i.attn_ln_1_w, i.attn_ln_1_b );
		ctx.add2( i.attnQuery, i.attn_q_w, i.attn_q_b );
		ctx.add( i.attnKey, i.attn_k_w );
		ctx.add2( i.attnValue, i.attn_v_w, i.attn_v_b );
		ctx.add2( i.crossAttnLn0, i.cross_attn_ln_0_w, i.cross_attn_ln_0_b );
		ctx.add2( i.crossAttnLn1, i.cross_attn_ln_1_w, i.cross_attn_ln_1_b );
		ctx.add2( i.crossAttnQuery, i.cross_attn_q_w, i.cross_attn_q_b );
		ctx.add2( i.mlpLn, i.mlp_ln_w, i.mlp_ln_b );
		ctx.add2( i.mlp0, i.mlp_0_w, i.mlp_0_b );
		ctx.add2( i.mlp1, i.mlp_1_w, i.mlp_1_b );
	}
	assert( ctx.isComplete() );
}
#endif