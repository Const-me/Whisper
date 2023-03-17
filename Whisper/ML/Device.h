#pragma once
#include <string>
#include "../D3D/sGpuInfo.h"
#include "LookupTables.h"
#include "DbgNanTest.h"

namespace DirectCompute
{
	struct Device
	{
		CComPtr<ID3D11Device> device;
		CComPtr<ID3D11DeviceContext> context;

		std::vector<CComPtr<ID3D11ComputeShader>> shaders;
		CComPtr<ID3D11Buffer> smallCb;
		sGpuInfo gpuInfo;
		LookupTables lookupTables;
#if DBG_TEST_NAN
		DbgNanTest nanTestBuffers;
#endif

		HRESULT create( uint32_t flags, const std::wstring& adapter );
		HRESULT createClone( const Device& source );
		void destroy();

		class ThreadSetupRaii
		{
			bool setup;
		public:
			ThreadSetupRaii( const Device* dev );
			~ThreadSetupRaii();
			ThreadSetupRaii( ThreadSetupRaii&& that ) noexcept
			{
				setup = that.setup;
				that.setup = false;
			}
			ThreadSetupRaii( const ThreadSetupRaii& ) = delete;
			void operator=( const ThreadSetupRaii& ) = delete;
		};

		ThreadSetupRaii setForCurrentThread() const
		{
			return ThreadSetupRaii{ this };
		}
	};
}