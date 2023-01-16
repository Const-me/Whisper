#pragma once
#include "ParallelForRunner.h"
#include "Tensor.h"

namespace CpuCompute
{
	HRESULT mulMat( Tensor& result, const Tensor& a, const Tensor& b, ParallelForRunner& pfor );
}

#if TENSOR_GGML_COMPAT
#include "../source/ggml.h"
inline HRESULT mulMat( ggml_tensor* result, const ggml_tensor* a, const ggml_tensor* b, CpuCompute::ParallelForRunner& pfor )
{
	CpuCompute::Tensor r{ result }, lhs{ a }, rhs{ b };
	return CpuCompute::mulMat( r, lhs, rhs, pfor );
}
#endif