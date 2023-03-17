#include "stdafx.h"
#include "LookupTables.h"
#include "../D3D/MappedResource.h"
#include "mlUtils.h"
#include "../D3D/shaders.h"
#include "../D3D/Binder.h"
#include "DbgNanTest.h"

namespace DirectCompute
{
	void zeroMemory( ID3D11UnorderedAccessView* uav, uint32_t length, bool fillWithNaN )
	{
		__m128i cbData = _mm_cvtsi32_si128( (int)length );
		cbData = _mm_insert_epi32( cbData, fillWithNaN ? 1 : 0, 1 );
		ID3D11Buffer* cb = updateSmallCb( cbData );

		ID3D11DeviceContext* ctx = context();
		ctx->CSSetUnorderedAccessViews( 0, 1, &uav, nullptr );
		csSetCB( cb );

		constexpr uint32_t THREADS = 512;
		constexpr uint32_t ITERATIONS = 128;
		constexpr uint32_t elementsPerGroup = THREADS * ITERATIONS;
		const uint32_t countGroups = ( length + elementsPerGroup - 1 ) / elementsPerGroup;
		bindShader( eComputeShader::zeroMemory );
		ctx->Dispatch( countGroups, 1, 1 );
	}

	void fillTensorWithNaN( ID3D11UnorderedAccessView* uav )
	{
		// Note we fill the complete unordered access view, ignoring the current size of the tensor. This is deliberate.
		D3D11_UNORDERED_ACCESS_VIEW_DESC desc;
		uav->GetDesc( &desc );
		assert( desc.Format == DXGI_FORMAT_R32_FLOAT || desc.Format == DXGI_FORMAT_R16_FLOAT );
		zeroMemory( uav, desc.Buffer.NumElements, true );
	}

	bool scanTensorForNaN( ID3D11ShaderResourceView* tensor, uint32_t length )
	{
#if DBG_TEST_NAN
		// Unlike fillTensorWithNaN function, this one only tests initial portion of the buffer
		const DbgNanTest& buffers = getNanTestBuffers();

		// Update constant buffer with elements = length, reset = true
		__m128i cbData = _mm_cvtsi32_si128( (int)length );
		cbData = _mm_insert_epi32( cbData, 1, 1 );
		ID3D11Buffer* cb = updateSmallCb( cbData );

		// Bind the compute shader and resources
		bindShader( eComputeShader::dbgFindNaN );
		csSetCB( cb );
		Binder bind;
		bind.bind( tensor, buffers );
		// Dispatch exactly 1 thread group of that shader
		context()->Dispatch( 1, 1, 1 );

		// Update constant buffer with elements = length, reset = false
		cbData = _mm_cvtsi32_si128( (int)length );
		updateSmallCb( cbData );

		// Dispatch the necessary count of that shader
		constexpr uint32_t THREADS = 512;
		constexpr uint32_t ITERATIONS = 128;
		constexpr uint32_t elementsPerGroup = THREADS * ITERATIONS;
		const uint32_t countGroups = ( length + elementsPerGroup - 1 ) / elementsPerGroup;
		context()->Dispatch( countGroups, 1, 1 );

		// Download result from GPU. This stalls the GPU pipeline, and ruins the performance.
		// This code better be disabled with DBG_TEST_NAN=0 macro
		return buffers.test();
#else
		return false;
#endif
	}
}