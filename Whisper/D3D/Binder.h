#pragma once
#include "device.h"

namespace DirectCompute
{
	class Binder
	{
		uint8_t maxSrv = 0;
		uint8_t maxUav = 0;

	public:
		Binder() = default;
		Binder( const Binder& ) = delete;

		void bind( ID3D11ShaderResourceView* srv0, ID3D11UnorderedAccessView* uav0 );
		void bind( ID3D11ShaderResourceView* srv0, ID3D11ShaderResourceView* srv1, ID3D11UnorderedAccessView* uav0 );
		void bind( std::initializer_list<ID3D11ShaderResourceView*> srvs, std::initializer_list<ID3D11UnorderedAccessView*> uavs );
		void bind( ID3D11UnorderedAccessView* uav0 );
		~Binder();
	};
}