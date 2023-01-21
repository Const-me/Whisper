#pragma once
#include "../D3D/device.h"
#include "DelayExecution.h"

namespace DirectCompute
{
	// A simple profiler which doesn't collect anything, used to measure time it took to load the model
	class GpuProfilerSimple
	{
		DelayExecution delay;
		CComPtr<ID3D11Query> disjoint, begin, end;
	public:
		HRESULT create();
		HRESULT time( uint64_t& rdi ) const;
	};
}