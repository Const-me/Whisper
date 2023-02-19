#pragma once

namespace DirectCompute
{
	// Update the small dynamic constant buffer
	ID3D11Buffer* __vectorcall updateSmallCb( __m128i cbData );
}