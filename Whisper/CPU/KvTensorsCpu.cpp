#include "stdafx.h"
#include "KvTensors.h"
using namespace CpuCompute;

// Create these two large tensors, FP16 precision
HRESULT KvTensors::create( const Whisper::sModelParams& mp )
{
	const uint32_t n_mem = mp.n_text_layer * mp.n_text_ctx;
	const uint32_t n_elements = mp.n_text_state * n_mem;

	const size_t cb = sizeof( uint16_t ) * (size_t)n_elements * 2;
	CHECK( memory.allocate( cb ) );

	uint16_t* pointer = (uint16_t*)memory.pointer();
	keys = pointer;
	values = pointer + n_elements;
	size = n_elements;
	return S_OK;
}