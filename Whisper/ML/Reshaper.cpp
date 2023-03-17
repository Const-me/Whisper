#include "stdafx.h"
#include "Reshaper.h"
#include "../D3D/MappedResource.h"
#include "../D3D/Binder.h"
#include "../D3D/shaders.h"
#include "reshapedMultiply.h"

namespace
{
	using namespace DirectCompute;
	struct Constants
	{
		// Size and strides of the source tensor
		TensorShape arg0;
		uint32_t zzPadding;
		// Count of elements per panel
		uint32_t panelSize;
		// Layer strides of the output matrix
		std::array<uint32_t, 2> layerStrides;
	};
}

HRESULT DirectCompute::Reshaper::createConstants()
{
	constexpr uint32_t cb = sizeof( Constants );
	CD3D11_BUFFER_DESC desc{ cb, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE };
	return device()->CreateBuffer( &desc, nullptr, &constantBuffer );
}

HRESULT DirectCompute::Reshaper::makePanels( Tensor& tensor, eDataType dataType )
{
	if( !constantBuffer )
		CHECK( createConstants() );

	constexpr uint32_t TILE_SIZE = ReshapedMultiply::TILE_SIZE;

	// Reshaping into column major horizontal panels, height = TILE_SIZE, width = width of the source matrix

	std::array<uint32_t, 4> ne = tensor.ne;
	const uint32_t groupsX = ( ne[ 1 ] + TILE_SIZE - 1 ) / TILE_SIZE;
	ne[ 1 ] = groupsX * TILE_SIZE;;
	// Each panel has [ size.x, TILE_SIZE ] elements
	const uint32_t panelSize = ne[ 0 ] * TILE_SIZE;
	
	Tensor result;
	result.create( dataType, ne, true );

	{
		MappedResource mapped;
		CHECK( mapped.map( constantBuffer, false ) );
		Constants& cb = *(Constants*)mapped.data();

		store( cb.arg0.ne, tensor.sizeVec() );
		store( cb.arg0.nb, tensor.stridesVec() );
		cb.panelSize = panelSize;
		cb.layerStrides[ 0 ] = result.nb[ 2 ];
		cb.layerStrides[ 1 ] = result.nb[ 3 ];
	}

	csSetCB( constantBuffer );
	{
		Binder bind;
		bind.bind( tensor, result );
		bindShader( eComputeShader::matReshapePanels );
		context()->Dispatch( groupsX, tensor.ne[ 2 ], tensor.ne[ 3 ] );
	}

	tensor.nb[ 0 ] = 0;
	tensor.nb[ 1 ] = panelSize;
	tensor.nb[ 2 ] = result.nb[ 2 ];
	tensor.nb[ 3 ] = result.nb[ 3 ];
	tensor.setGpuViews( result );
	return S_OK;
}

DirectCompute::Reshaper::~Reshaper()
{
	if( constantBuffer )
		csSetCB( nullptr );
}