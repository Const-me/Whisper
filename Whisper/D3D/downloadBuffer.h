#pragma once

namespace DirectCompute
{
	// Download a buffer from VRAM into std::vector
	// The function is relatively expensive, creates a temporary staging buffer on each call, and only used to test things.
	template<class E>
	HRESULT downloadBuffer( ID3D11ShaderResourceView* srv, std::vector<E>& vec );
}