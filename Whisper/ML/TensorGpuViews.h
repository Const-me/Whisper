#pragma once
#include <stdint.h>
#include "../D3D/device.h"

namespace DirectCompute
{
	class TensorGpuViews
	{
	protected:
		CComPtr<ID3D11ShaderResourceView> srv;
		CComPtr<ID3D11UnorderedAccessView> uav;

	public:

		operator ID3D11ShaderResourceView* ( ) const { return srv; }
		operator ID3D11UnorderedAccessView* ( ) const { return uav; }

		HRESULT create( ID3D11Buffer* buffer, DXGI_FORMAT format, size_t countElements, bool makeUav );

		void clear()
		{
			srv = nullptr;
			uav = nullptr;
		}

		void setGpuViews( ID3D11ShaderResourceView* read, ID3D11UnorderedAccessView* write = nullptr )
		{
			srv = read;
			uav = write;
		}
	};
}