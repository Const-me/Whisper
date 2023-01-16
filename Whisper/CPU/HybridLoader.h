#pragma once
#include "DecoderTensors.h"
#include <atlstr.h>
#include <atlcoll.h>
#include "../../ComLightLib/streams.h"

namespace CpuCompute
{
	__interface iLoaderProgressSink
	{
		HRESULT gotBytes( int64_t cb );
	};

	class HybridLoader
	{
		DecoderTensors& destination;
		CAtlMap<CStringA, Tensor*> map;
		size_t bufferBytes = 0;

		struct alignas( 32 ) PendingTensor
		{
			Tensor* destPointer = nullptr;
			int64_t streamOffset = 0;
			size_t bufferOffset = 0;
			size_t payloadBytes = 0;
		};
		std::vector<PendingTensor> pending;

	public:

		HybridLoader( DecoderTensors& m, int countLayers );

		HRESULT setupTensor( const CStringA& name, int n_dims, int ftype, const std::array<int, 4>& ne, ComLight::iReadStream* stream, int64_t& postponedBytes );

		HRESULT completeLoad( ComLight::iReadStream* stream, iLoaderProgressSink& progressSink );
	};
}