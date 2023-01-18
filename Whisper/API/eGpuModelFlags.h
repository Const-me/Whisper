#pragma once
#include <stdint.h>

namespace Whisper
{
	enum struct eGpuModelFlags : uint32_t
	{
		Wave32 = 1,
		Wave64 = 2,
		NoReshapedMatMul = 4,
		UseReshapedMatMul = 8,
	};
}