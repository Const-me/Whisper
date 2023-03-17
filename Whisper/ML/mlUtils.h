#pragma once

namespace DirectCompute
{
	// Update the small dynamic constant buffer
	ID3D11Buffer* __vectorcall updateSmallCb( __m128i cbData );

	// Fill the tensor with either 0.0 or NaN values
	void zeroMemory( ID3D11UnorderedAccessView* uav, uint32_t length, bool fillWithNaN = false );

	// Fill the complete UAV with NaN values
	void fillTensorWithNaN( ID3D11UnorderedAccessView* uav );

	// true when the tensor contains at least 1 NaN value
	bool scanTensorForNaN( ID3D11ShaderResourceView* tensor, uint32_t length );

	// Create SRV on another device, reusing the resource
	HRESULT cloneResourceView( ID3D11ShaderResourceView* rsi, ID3D11ShaderResourceView** rdi );
}