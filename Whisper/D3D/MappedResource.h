#pragma once
#include "device.h"
#include <assert.h>

namespace DirectCompute
{
	class MappedResource
	{
		D3D11_MAPPED_SUBRESOURCE mapped;
		ID3D11Resource* resource;
	public:
		MappedResource();
		HRESULT map( ID3D11Resource* res, bool reading );
		~MappedResource();

		void* data() const
		{
			assert( nullptr != mapped.pData );
			return mapped.pData;
		}
	};
}