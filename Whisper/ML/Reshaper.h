#pragma once
#include "Tensor.h"

namespace DirectCompute
{
	// This class reshapes some of the model’s tensor, immediately after they’re loaded.
	// That feature is used on all AMD GPUs.
	class Reshaper
	{
		CComPtr<ID3D11Buffer> constantBuffer;
		HRESULT createConstants();

	public:
		~Reshaper();
		HRESULT makePanels( Tensor& tensor, eDataType dataType );
	};
}