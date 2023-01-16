#include "stdafx.h"
#include "tensorOpsTests.h"
#include "MlContext.h"
#include "TensorEx.h"
#include "../D3D/shaders.h"
#include "../D3D/Binder.h"
#include "testUtils.h"
#include "../Whisper/WhisperContext.h"

void DirectCompute::testMulMat( const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst, const void* tempBuffer )
{
	return;
	CaptureRaii capture;
	const size_t nb00 = src0->nb[ 0 ];
	const size_t nb01 = src0->nb[ 1 ];

	if( src0->type != GGML_TYPE_F16 )
		return; // TODO

	if( nb01 < nb00 )
		return;	// TODO

	WhisperContext& ctx = WhisperContext::current();

	Tensor arg0, arg1;
	check( arg0.create( *src0, eBufferUse::Immutable, true ) );
	check( arg1.create( *src1, eBufferUse::Immutable, true ) );
	TensorEx res;
	check( res.create( *dst, eBufferUse::ReadWriteDownload, false ) );

	ctx.mulMat( arg0, arg1, res );

	std::vector<float> tv;
	check( res.download( tv ) );

	const size_t len = tv.size();
	computeDiff( tv.data(), (const float*)dst->data, len ).print( "testMulMat-product" );

#if 0
	dbgWriteBinaryFile( L"product-orig.bin", dst->data, len * 4 );
	dbgWriteBinaryFile( L"product-gpu.bin", tv.data(), len * 4 );
	__debugbreak();
#endif
}

#if 0
void DirectCompute::testMulMatReshape( const ggml_tensor* src1, const void* tempBuffer )
{
	Tensor src;
	check( src.create( *src1, eBufferUse::Immutable, true ) );

	const size_t ne10 = (uint32_t)src1->ne[ 0 ];
	const size_t ne11 = (uint32_t)src1->ne[ 1 ];
	const size_t ne12 = (uint32_t)src1->ne[ 2 ];
	const size_t ne13 = (uint32_t)src1->ne[ 3 ];
	if( 1 != ne13 )
		throw E_UNEXPECTED;
	const size_t tempLength = ne10 * ne11 * ne12 * ne13;

	Context& ctx = Context::current();
	const ReadWriteViews& temp = ctx.temp.fp16( tempLength );

	{
		Binder bind;
		ctx.cb.bind();

		bindShader( eComputeShader::mulMatDotReshape );

		ctx.cb.update( src );
		bind.bind( src, temp );
		context()->Dispatch( (UINT)ne11, (UINT)ne12, 1 );
	}

	std::vector<uint16_t> reshaped;
	check( downloadBuffer( temp, reshaped ) );
	computeDiff( reshaped.data(), (const uint16_t*)tempBuffer, reshaped.size() ).print( "testMulMatReshape" );

#if 0
	dbgWriteBinaryFile( L"fp32.bin", src1->data, ggml_nbytes( src1 ) );
	dbgWriteBinaryFile( L"fp16-cpu.bin", tempBuffer, reshaped.size() * 2 );
	dbgWriteBinaryFile( L"fp16-gpu.bin", reshaped.data(), reshaped.size() * 2 );
	__debugbreak();
#endif
}
#endif

void DirectCompute::computeMulMat( const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst )
{
	CaptureRaii capture;
	const size_t nb00 = src0->nb[ 0 ];
	const size_t nb01 = src0->nb[ 1 ];

	if( src0->type != GGML_TYPE_F16 )
		throw E_INVALIDARG;
	if( nb01 < nb00 )
		throw E_INVALIDARG;

	WhisperContext& ctx = WhisperContext::current();

	Tensor arg0, arg1;
	check( arg0.create( *src0, eBufferUse::Immutable, true ) );
	check( arg1.create( *src1, eBufferUse::Immutable, true ) );
	TensorEx res;
	check( res.create( *dst, eBufferUse::ReadWriteDownload, false ) );

	ctx.mulMat( arg0, arg1, res );

	check( res.download( dst->data ) );
}

void DirectCompute::testFlashAttention( const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v, bool masked, const ggml_tensor* dst )
{
	CaptureRaii capture;

	Tensor Q, K, V;
	TensorEx res;
	check( Q.create( *q, eBufferUse::Immutable, true ) );
	check( K.create( *k, eBufferUse::Immutable, true ) );
	check( V.create( *v, eBufferUse::Immutable, true ) );
	check( res.create( *dst, eBufferUse::ReadWriteDownload, false ) );

	WhisperContext& ctx = WhisperContext::current();
	ctx.flashAttention( Q, K, V, res, masked );

	std::vector<float> tv;
	check( res.download( tv ) );

	const size_t len = tv.size();
	computeDiff( tv.data(), (const float*)dst->data, len ).print( "testFlashAttention" );
}

void DirectCompute::computeFlashAttention( const ggml_tensor* q, const ggml_tensor* k, const ggml_tensor* v, bool masked, ggml_tensor* dst )
{
	CaptureRaii capture;

	Tensor Q, K, V;
	TensorEx res;
	check( Q.create( *q, eBufferUse::Immutable, true ) );
	check( K.create( *k, eBufferUse::Immutable, true ) );
	check( V.create( *v, eBufferUse::Immutable, true ) );
	check( res.create( *dst, eBufferUse::ReadWriteDownload, false ) );

	WhisperContext& ctx = WhisperContext::current();
	ctx.flashAttention( Q, K, V, res, masked );

	check( res.download( dst->data ) );
}

void DirectCompute::testConvolution( const ggml_tensor* src0, const ggml_tensor* src1, const ggml_tensor* dst )
{
	CaptureRaii capture;

	Tensor arg0, arg1;
	check( arg0.create( *src0, eBufferUse::Immutable, true ) );
	check( arg1.create( *src1, eBufferUse::Immutable, true ) );
	TensorEx res;
	check( res.create( *dst, eBufferUse::ReadWriteDownload, false ) );

	WhisperContext& ctx = WhisperContext::current();
	ctx.convolution( arg0, arg1, res );

	std::vector<float> tv;
	check( res.download( tv ) );

	const size_t len = tv.size();
	computeDiff( tv.data(), (const float*)dst->data, len ).print( "testConvolution" );
}

void DirectCompute::computeConvolution( const ggml_tensor* src0, const ggml_tensor* src1, ggml_tensor* dst )
{
	CaptureRaii capture;

	Tensor arg0, arg1;
	check( arg0.create( *src0, eBufferUse::Immutable, true ) );
	check( arg1.create( *src1, eBufferUse::Immutable, true ) );
	TensorEx res;
	check( res.create( *dst, eBufferUse::ReadWriteDownload, false ) );

	WhisperContext& ctx = WhisperContext::current();
	ctx.convolution( arg0, arg1, res );

	res.download( dst->data );
}