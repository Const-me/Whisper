#pragma once
#include <vector>
#include "TempBuffers.h"
#include "ConstantBuffer.h"
#include "Tensor.h"
#include "../Utils/GpuProfiler.h"
#include "../Utils/ProfileCollection.h"

namespace DirectCompute
{
	enum struct eComputeShader : uint16_t;

	class MlContext
	{
		// When false, the implementation is 100% compatible with the CPU-running code written by Georgi Gerganov
		// When true, the implementation is much faster, and doesn't require FP64 support in the compute shaders.
		// FP64 is an optional feature, not all GPUs support that.
		static constexpr bool enableInexactOptimizations = true;

		ConstantBuffer cb;
		TempBuffers temp;
		CComPtr<ID3D11Buffer> flashAttentionConstants;

		void convolutionImpl( const Tensor& a, const Tensor& b, Tensor& res, bool is2 );

		void cwiseBinary( const Tensor& a, const Tensor& b, Tensor& res, eComputeShader cs );
		Tensor cwiseBinary( const Tensor& a, const Tensor& b, eComputeShader cs );

		void mulMatDot( const Tensor& a, const Tensor& b, Tensor& res );
		void mulMatMad( const Tensor& a, const Tensor& b, Tensor& res );
		void mulMatTiled( const Tensor& a, const Tensor& b, Tensor& res );

		void bindShader( eComputeShader cs );

	protected:
		void copyImpl( const Tensor& a, Tensor& res, bool downcastFp32 );

		// Create a dense output tensor for the results of a computation
		// Override this method to implement a pool of these tensors
		virtual Tensor createTensor( eDataType type, const std::array<uint32_t, 4>& ne );

		Tensor createTensor( eDataType type, std::initializer_list<uint32_t> ne );

		GpuProfiler profiler;

	public:
		MlContext( Whisper::ProfileCollection& profileColl );
		MlContext( const MlContext& ) = delete;

		// res = a * b
		void mulMat( const Tensor& a, const Tensor& b, Tensor& res );

		void flashAttention( const Tensor& q, const Tensor& k, const Tensor& v, Tensor& res, bool masked );

		inline void convolution( const Tensor& a, const Tensor& b, Tensor& res )
		{
			convolutionImpl( a, b, res, false );
		}
		void convolution2( const Tensor& a, const Tensor& b, Tensor& res )
		{
			convolutionImpl( a, b, res, true );
		}

		void norm( const Tensor& a, Tensor& res );

		Tensor conv_1d_1s( const Tensor& a, const Tensor& b );
		Tensor conv_1d_2s( const Tensor& a, const Tensor& b );

		Tensor add( const Tensor& a, const Tensor& b );
		void addInPlace( Tensor& a, const Tensor& b );

		Tensor view2d( const Tensor& a, uint32_t ne0, uint32_t ne1, uint32_t nb1, uint32_t offset );
		Tensor transpose( const Tensor& a );

		Tensor norm( const Tensor& a );
		Tensor mulMat( const Tensor& a, const Tensor& b );
		Tensor mulMatEx( const Tensor& a, const Tensor& b, const char* tagName );
		Tensor permute( const Tensor& a, uint8_t axis0, uint8_t axis1, uint8_t axis2, uint8_t axis3 );
		Tensor flashAttention( const Tensor& q, const Tensor& k, const Tensor& v, bool masked );

		Tensor copy( const Tensor& a, eDataType type, std::initializer_list<uint32_t> size );
		void copyInPlace( Tensor& dest, const Tensor& a, eDataType type, std::initializer_list<uint32_t> size );

		void dbgPrintDifference( const ggml_tensor* reference, const Tensor& gpu, const char * what, bool trapToDebugger = true );

		void scale( Tensor& a, float mul );

		void addRepeat( Tensor& a, const Tensor& b );
		void addRepeatScale( Tensor& a, const Tensor& b, float scale );
		void fmaRepeat( Tensor& a, const Tensor& mul, const Tensor& add );

		// ggml_diag_mask_inf
		void diagMaskInf( Tensor& a, uint32_t n_past );
		// ggml_soft_max
		void softMax( Tensor& a, float inputScale = 1.0f );

		void addRepeatGelu( Tensor& a, const Tensor& b );

		// Extract rows from tokenEmbedding matrix, row indices are taken from the `embd` R32_UINT row vector
		// Extract same count of rows from positionalEmbedding matrix, starting at the `pastTokensCount` row
		// Return a new FP32 matrix with the sum of these rows
		Tensor addRows( const Tensor& tokenEmbedding, const Tensor& positionalEmbedding, const Tensor& embd, uint32_t pastTokensCount );

		Tensor reshapePanels( const Tensor& a );

		Tensor mulMatTiledEx( const Tensor& a, const Tensor& b );
		Tensor mulMatByRowTiledEx( const Tensor& a, const Tensor& b );

		// An equivalent of addRepeat( dest, pattern ) followed by addInPlace( dest, finalAdd )
		void addRepeatEx( Tensor& dest, const Tensor& pattern, const Tensor& finalAdd );

		__m128i getMemoryUse() const;
	};
}