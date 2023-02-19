#pragma once
#include "TraceWriter.h"
#include "../../ML/mlUtils.h"

namespace Tracing
{
#if SAVE_DEBUG_TRACE
	void traceCreate( LPCTSTR path );
	void traceClose();

	iTraceWriter* getWriter();

	inline HRESULT tensor( const ItemName& name, const DirectCompute::Tensor& tensor )
	{
		iTraceWriter* w = getWriter();
		if( w )
			return w->tensor( name, tensor );
		return S_FALSE;
	}
	inline HRESULT tensor( const ItemName& name, const CpuCompute::Tensor& tensor )
	{
		iTraceWriter* w = getWriter();
		if( w )
			return w->tensor( name, tensor );
		return S_FALSE;
	}

	inline HRESULT tensor( const ItemName& name, const ggml_tensor* tensor )
	{
		iTraceWriter* w = getWriter();
		if( w )
			return w->tensor( name, *tensor );
		return S_FALSE;
	}

	void delayTensor( const ItemName& name, const ggml_tensor* tensor );
	HRESULT writeDelayedTensors();

	inline HRESULT buffer( const ItemName& name, const void* rsi, size_t length, eDataType dt )
	{
		iTraceWriter* w = getWriter();
		if( w )
			return w->buffer( name, rsi, length, dt );
		return S_FALSE;
	}

	inline HRESULT vector( const ItemName& name, const std::vector<float>& vec )
	{
		const float* rsi = vec.empty() ? nullptr : vec.data();
		return buffer( name, rsi, vec.size(), eDataType::FP32 );
	}
	inline HRESULT vector( const ItemName& name, const float* rsi, size_t length )
	{
		return buffer( name, rsi, length, eDataType::FP32 );
	}
#else
	inline void traceCreate( LPCTSTR path ) { }
	inline void traceClose() { }
#if DBG_TEST_NAN
	HRESULT tensor( const ItemName& name, const DirectCompute::Tensor& tensor );
#else
	inline HRESULT tensor( const ItemName& name, const DirectCompute::Tensor& tensor ) { return S_FALSE; }
#endif
	inline HRESULT tensor( const ItemName& name, const CpuCompute::Tensor& tensor ) { return S_FALSE; }
	inline HRESULT tensor( const ItemName& name, const ggml_tensor* tensor ) { return S_FALSE; }
	inline HRESULT buffer( const ItemName& name, const void* rsi, size_t length, eDataType dt ) { return S_FALSE; }
	inline HRESULT vector( const ItemName& name, const std::vector<float>& vec ) { return S_FALSE; }
	inline void delayTensor( const ItemName& name, const ggml_tensor* tensor ) { }
	inline HRESULT writeDelayedTensors() { return S_FALSE; }
	inline HRESULT vector( const ItemName& name, const float* rsi, size_t length ) { }
#endif
}