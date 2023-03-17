#pragma once
#include <stdint.h>

namespace DirectCompute
{
	// DXGI_ADAPTER_DESC.VendorId magic numbers; they come from that database: https://pcisig.com/membership/member-companies
	enum struct eGpuVendor : uint16_t
	{
		AMD = 0x1002,
		NVidia = 0x10de,
		Intel = 0x8086,
		VMWare = 0x15ad,
	};

	enum struct eGpuEffectiveFlags : uint8_t
	{
		Wave64 = 1,
		ReshapedMatMul = 2,
		Cloneable = 4,
	};

	struct sGpuInfo
	{
		eGpuEffectiveFlags flags;
		eGpuVendor vendor;
		uint16_t device, revision;
		uint32_t subsystem;
		size_t vramDedicated, ramDedicated, ramShared;
		std::wstring description;

		inline bool wave64() const
		{
			return 0 != ( (uint8_t)flags & (uint8_t)eGpuEffectiveFlags::Wave64 );
		}

		// On nVidia 1080Ti that approach is much slower, by a factor of 2.4
		// On AMD Cezanne that approach is faster by a factor of 0.69, i.e. 30% faster.
		// Dunno why that is, maybe 'coz on that AMD complete panels fit in L3 cache.
		// Anyway, we do want extra 30% perf on AMD Cezanne, so only using that code on AMD GPUs.
		// Dunno how it gonna behave on other GPUs, need to test.
		inline bool useReshapedMatMul() const
		{
			return 0 != ( (uint8_t)flags & (uint8_t)eGpuEffectiveFlags::ReshapedMatMul );
		}

		inline bool cloneableModel() const
		{
			return 0 != ( (uint8_t)flags & (uint8_t)eGpuEffectiveFlags::Cloneable );
		}
	};
}