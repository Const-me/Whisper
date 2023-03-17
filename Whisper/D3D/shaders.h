#pragma once
#include "shaderNames.h"

namespace DirectCompute
{
	HRESULT createComputeShaders( std::vector<CComPtr<ID3D11ComputeShader>>& shaders );

	void bindShader( eComputeShader shader );
}