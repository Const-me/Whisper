#pragma once
#include <stdint.h>

namespace Whisper
{
	enum struct eModelImplementation : uint32_t
	{
		// GPGPU implementation based on Direct3D 11.0 compute shaders
		GPU = 1,

		// A hybrid implementation which uses DirectCompute for encode, and decodes on CPU
		// Not implemented in the published builds of the DLL. To enable, change BUILD_HYBRID_VERSION macro to 1
		Hybrid = 2,

		// A reference implementation which uses the original GGML CPU-running code
		// Not implemented in the published builds of the DLL. To enable, change BUILD_BOTH_VERSIONS macro to 1
		Reference = 3,
	};

	enum struct eGpuModelFlags : uint32_t
	{
		Wave32 = 1,
		Wave64 = 2,
		NoReshapedMatMul = 4,
		UseReshapedMatMul = 8,
		Cloneable = 0x10,
	};

	struct sModelSetup
	{
		eModelImplementation impl = eModelImplementation::GPU;
		uint32_t flags = 0;
		const wchar_t* adapter = nullptr;
	};

	// Function pointer to enumerate GPUs
	using pfnListAdapters = void( __stdcall* )( const wchar_t* name, void* pv );

	// Function pointer to receive array of tokens from iModel.tokenize() API method
	using pfnDecodedTokens = void( __stdcall* )( const int* tokens, int tokensLength, void* pv );
}