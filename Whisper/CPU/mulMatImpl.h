#pragma once
// Matrix*matrix multiplication is the most expensive algorithm in the model, by far.
// For this reason, the code in this source file, and in the mulMat.kernel.hpp header, is optimized for performance. Readability suffers.
// The implementation is inspired by following two articles:
// https://gist.github.com/nadavrot/5b35d44e8ba3dd718e595e40184d03f0
// https://link.springer.com/article/10.1007/s11227-022-05003-3
#include "ParallelForRunner.h"
#include "Tensor.h"

namespace CpuCompute
{
	// Abstract base class for all implementations, to reduce binary size
	class MulMatBase : public iComputeRange
	{
	protected:
		// Pointers to the payload of the output matrix
		float* const resultPointer;

		// Lengths of the dot products to compute, equal to width of both source matrices
		uint32_t length;

		// Last 3 strides of the output matrix, expressed as count of elements. The first one is always 1 because the output matrix is continuous.
		std::array<uint32_t, 3> resultStrides;

		// Size of the output matrix
		std::array<uint32_t, 4> resultSize;

		// Pointers to the payload of the source matrices
		const void* const pa;
		const void* const pb;

		// Matrix strides, expressed as count of elements
		std::array<uint32_t, 4> stridesA, stridesB;

		// Total count of panels in the layer of the output matrix.
		// The last panel might be incomplete, with smaller height.
		// The thread-local buffer however is always complete, unused elements will be zeros.
		uint32_t countPanels;

		// Complete tiles in the length of the panel
		uint32_t completeTilesPerPanel;

		// Count of the last remainder columns in the panel, can be 0
		uint8_t lastColumnsInPanel;

		// Same as panelHeightRegs template argument - height of the panels, in AVX vectors
		uint8_t panelHeightRegisters;

		// Same as tileWidthFloats template argument - width of the tile, in floats
		uint8_t tileWidth;

		// Method pointer to reshape a panel from the source matrix into a thread-local buffer
		using pfnTransposePanel = HRESULT( MulMatBase::* )( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const;
		pfnTransposePanel pfnMakePanel;
		// The object which implements multithreading for this job, and supplies memory for thread-local buffers
		ParallelForRunner& runner;

		// Count of FP16 values in the thread-local panel buffer
		uint32_t floatsPerPanel() const
		{
			return length * panelHeightRegisters * 8;
		}

		// Transpose a horizontal panel of the first matrix, when the rows are continuous in that matrix
		HRESULT transposePanel( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const;
		HRESULT transposePanelAvx2( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const;
		// Copy a horizontal panel of the first matrix without transpose, for column major layout of that matrix
		HRESULT copyPanelColumnMajor8( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const;
		HRESULT copyPanelColumnMajor16( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const;
		HRESULT copyPanelColumnMajor32( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const;
		// Transpose a panel of the first matrix for irregular layout of that matrix, when neither rows nor columns are at sequential addresses.
		// This one ain't implemented yet.
		HRESULT gatherPanel( uint16_t* rdi, size_t i, size_t m2, size_t m3 ) const;

		const uint16_t* getPanelA( size_t i, size_t m2, size_t m3 ) const;
		// Pointer to the first element of the second source matrix in the specified layer
		const float* getLayerB( size_t m2, size_t m3 ) const;

		// Pointer to the first element of the output tile of the result matrix
		float* getPanelDest( size_t i, size_t m2, size_t m3 ) const
		{
			float* rdi = resultPointer;
			rdi += m2 * resultStrides[ 1 ];
			rdi += m3 * resultStrides[ 2 ];
			rdi += i * panelHeightRegisters * 8;
			return rdi;
		}

		static const bool haveAvx2;
	public:
		MulMatBase( Tensor& result, const Tensor& a, const Tensor& b, ParallelForRunner& pfor, uint8_t panelHeightRegs, uint8_t tileWidthFloats );
		HRESULT run( ParallelForRunner& pfor );
	};

	// This class actually contains the kernels implementations
	template<uint8_t panelHeightRegs, uint8_t tileWidthFloats>
	class MulMatImpl : public MulMatBase
	{
		HRESULT __stdcall compute( size_t i, size_t end ) const noexcept override final;

	public:
		MulMatImpl( Tensor& result, const Tensor& a, const Tensor& b, ParallelForRunner& pfor ) :
			MulMatBase( result, a, b, pfor, panelHeightRegs, tileWidthFloats )
		{ }
	};
}