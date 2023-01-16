#pragma once
#include "../source/ggml.h"

namespace DirectCompute
{
	// void testMulMatReshape( const ggml_tensor* src1, const void* tempBuffer );
	void testMulMat( const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst, const void* tempBuffer );
	void computeMulMat( const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst );

	void testFlashAttention( const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v, bool masked, const ggml_tensor* dst );
	void computeFlashAttention( const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v, bool masked, ggml_tensor* dst );

	void testConvolution( const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst );
	void computeConvolution( const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst );
}