#include "stdafx.h"
#include <intrin.h>
#include "mulMatImpl.h"
#include "mulMat.kernel.hpp"

#define DBG_TRACK_TEMPLATE_INSTANTIATION 0

#if DBG_TRACK_TEMPLATE_INSTANTIATION
#include <unordered_set>
static std::unordered_set<uint16_t> g_mulMatTemplates;
#endif

namespace
{
	using namespace CpuCompute;

	bool checkAvx2Support()
	{
		int cpuInfo[ 4 ];
		__cpuid( cpuInfo, 7 );
		return ( cpuInfo[ 1 ] & ( 1 << 5 ) ) != 0;
	}

	// a / b, rounded up to the next integer
	inline uint32_t divRoundUp( uint32_t a, uint32_t b )
	{
		assert( b != 0 );
		return ( a + ( b - 1 ) ) / b;
	}
}

const bool MulMatBase::haveAvx2 = checkAvx2Support();

MulMatBase::MulMatBase( Tensor& result, const Tensor& a, const Tensor& b, ParallelForRunner& pfor, uint8_t panelHeightRegs, uint8_t tileWidthFloats ) :
	resultPointer( result.fp32() ),
	pa( a.data() ),
	pb( b.data() ),
	runner( pfor )
{
	length = a.ne[ 0 ];
	resultStrides[ 0 ] = result.nb[ 1 ];
	resultStrides[ 1 ] = result.nb[ 2 ];
	resultStrides[ 2 ] = result.nb[ 3 ];
	store( resultSize, result.sizeVec() );
	store( stridesA, a.stridesVec() );
	store( stridesB, b.stridesVec() );

	countPanels = divRoundUp( resultSize[ 0 ], panelHeightRegs * 8 );
	completeTilesPerPanel = resultSize[ 1 ] / tileWidthFloats;
	lastColumnsInPanel = (uint8_t)( resultSize[ 1 ] % tileWidthFloats );
	this->panelHeightRegisters = panelHeightRegs;
	this->tileWidth = tileWidthFloats;

	// Pick a method which reshapes a panel of the matrix A into the shape we need to compute the product
	// Store the pointer to that method in the field of this class
	if( a.nb[ 0 ] == 1 )
	{
		if( haveAvx2 )
			pfnMakePanel = &MulMatBase::transposePanelAvx2;
		else
			pfnMakePanel = &MulMatBase::transposePanel;
	}
	else if( a.nb[ 1 ] == 1 )
	{
		switch( panelHeightRegs )
		{
		case 1:
			pfnMakePanel = &MulMatBase::copyPanelColumnMajor8;
			break;
		case 2:
			pfnMakePanel = &MulMatBase::copyPanelColumnMajor16;
			break;
		case 4:
			pfnMakePanel = &MulMatBase::copyPanelColumnMajor32;
			break;
		default:
			throw E_NOTIMPL;
		}
	}
	else
		pfnMakePanel = &MulMatBase::gatherPanel;

	// That last version is generic and very simple, unlikely to have weird bugs
	// pfnMakePanel = &MulMatBase::gatherPanel;

#if DBG_TRACK_TEMPLATE_INSTANTIATION
	uint16_t key = panelHeightRegs;
	key = key << 8;
	key |= tileWidthFloats;
	if( !g_mulMatTemplates.emplace( key ).second )
		return;
	logDebug( u8"MulMatImpl<panelHeightRegs = %i, tileWidthFloats = %i>", (int)panelHeightRegs, (int)tileWidthFloats );
#endif
}

HRESULT MulMatBase::run( ParallelForRunner& pfor )
{
	size_t length = (size_t)countPanels * resultSize[ 2 ] * resultSize[ 3 ];
	return pfor.parallelFor( *this, length );
}

const float* MulMatBase::getLayerB( size_t m2, size_t m3 ) const
{
	const float* rsi = (const float*)this->pb;
	rsi += m2 * stridesB[ 2 ];
	rsi += m3 * stridesB[ 3 ];
	return rsi;
}

// This method is the main one, it’s called by the thread pool
template<uint8_t panelHeightRegs, uint8_t tileWidthFloats>
HRESULT __stdcall MulMatImpl<panelHeightRegs, tileWidthFloats>::compute( size_t i, size_t end ) const noexcept
{
	// Allocate a thread-local buffer for the transposed panel
	constexpr size_t panelHeightFloats = panelHeightRegs * 8;
	uint16_t* const panel = (uint16_t*)runner.threadLocalBuffer( floatsPerPanel() * 2 );
	const size_t resultStride = resultStrides[ 0 ];

	// Load a few numbers from this class into local variables, while upcasting from DWORD into size_t
	const size_t length = this->length;
	const std::array<size_t, 2> stridesB{ this->stridesB[ 0 ], this->stridesB[ 1 ] };

	// This outer loop iterates over the panels assigned to the current thread
	// For example, matrix A of size [ 1024, 1024 ] may be split into panels of size [ 1024, 16 ]
	// Each iteration of that loop computes matrix product of that panel, with the complete matrix B
	for( ; i < end; i++ )
	{
		const size_t iPanel = i % countPanels;
		size_t j = i / countPanels;
		const size_t m2 = j % (size_t)resultSize[ 2 ];
		const size_t m3 = j / (size_t)resultSize[ 2 ];

		CHECK( ( this->*pfnMakePanel )( panel, iPanel, m2, m3 ) );
		// We got a column-major panel in the thread local buffer, of size [ length, panelHeightRegs * 8 ]
		// Hopefully, these buffers should all fit at least in L3 cache
		// The longest matrix I saw in the debugger had 4096 elements, with panelHeightRegs = 4 that's 256 kb of data in the panel
		const float* pb = getLayerB( m2, m3 );
		float* rdi = getPanelDest( iPanel, m2, m3 );

		const size_t storeWidth = std::min( panelHeightFloats, (size_t)resultSize[ 0 ] - iPanel * panelHeightFloats );
		std::array<__m256, panelHeightRegs> vecPanel;
#if 1
		ResultTile<panelHeightRegs, tileWidthFloats> tile;

		// This loop iterates over tiles within the panel.
		// Each iteration of the loop computes an output tile of the result matrix.
		for( j = 0; j < completeTilesPerPanel; j++, pb += tileWidthFloats * stridesB[ 1 ], rdi += resultStride * tileWidthFloats )
		{
			setZero( tile.arr );
			const uint16_t* rsiA = panel;
			const uint16_t* const rsiAEnd = panel + length * panelHeightFloats;
			const float* rsiB = pb;
			// This loop runs for `length` iterations, iterates over the first dimensions of both matrices, accumulating these dot products we're after
			for( ; rsiA < rsiAEnd; rsiA += panelHeightFloats, rsiB += stridesB[ 0 ] )
			{
				loadPanel( rsiA, vecPanel );
				tile.kernel( vecPanel, rsiB, stridesB[ 1 ] );
			}
			tile.store( rdi, storeWidth, tileWidthFloats, resultStride );
		}

		if( 0 != lastColumnsInPanel )
		{
			setZero( tile.arr );
			const uint16_t* rsiA = panel;
			const uint16_t* rsiAEnd = panel + length * panelHeightFloats;
			const float* rsiB = pb;
			for( ; rsiA < rsiAEnd; rsiA += panelHeightFloats, rsiB += stridesB[ 0 ] )
			{
				loadPanel( rsiA, vecPanel );
				tile.kernelPartial( vecPanel, rsiB, stridesB[ 1 ], lastColumnsInPanel );
			}
			tile.store( rdi, storeWidth, lastColumnsInPanel, resultStride );
		}
#else
		// This version bypasses horizontal tiling, instead implements a brute force algorithm to multiply the current panel by the complete B matrix
		// Not terribly efficient, only implemented for debugging purposes
		const size_t resHeight = resultSize[ 1 ];
		std::array<__m256, panelHeightRegs> tile;
		for( size_t j = 0; j < resHeight; j++, pb += stridesB[ 1 ], rdi += resultStride )
		{
			setZero( tile );

			const uint16_t* rsiA = panel;
			const uint16_t* const rsiAEnd = panel + length * panelHeightFloats;
			const float* rsiB = pb;
			for( size_t k = 0; k < length; k++, rsiA += panelHeightFloats, rsiB += stridesB[ 0 ] )
			{
				loadPanel( rsiA, vecPanel );
				const __m256 b = _mm256_broadcast_ss( rsiB );
				for( size_t r = 0; r < panelHeightRegs; r++ )
					tile[ r ] = _mm256_fmadd_ps( vecPanel[ r ], b, tile[ r ] );
			}

			alignas( 32 ) std::array<float, panelHeightFloats> arr;
			for( size_t k = 0; k < panelHeightRegs; k++ )
				_mm256_store_ps( &arr[ k * 8 ], tile[ k ] );
			memcpy( rdi, arr.data(), storeWidth * 4 );
		}
#endif
	}
	return S_OK;
}

// Instantiate the templates we need
template class MulMatImpl<4, 1>;
template class MulMatImpl<1, 1>;
template class MulMatImpl<4, 2>;
template class MulMatImpl<1, 2>;
template class MulMatImpl<2, 3>;
template class MulMatImpl<1, 3>;
template class MulMatImpl<2, 4>;
template class MulMatImpl<1, 4>;