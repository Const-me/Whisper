#pragma once
#include "ModelBuffers.h"
#include <map>

namespace DirectCompute
{
	struct ModelLoader
	{
		ModelLoader( int encoderLayers, int decoderLayers );

		void add( const ggml_tensor* ggml, Tensor& gpu );

		void add( const ggml_tensor* w, const ggml_tensor* b, TensorPair& gpu )
		{
			add( w, gpu.w );
			add( b, gpu.b );
		}

		bool tryLoad( const ggml_tensor* ggml );

		ModelBuffers& model;

	private:

		Tensor* lookup( const ggml_tensor* ggml ) const;

		std::map<const ggml_tensor*, Tensor*> map;
	};
}