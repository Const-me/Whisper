#include "stdafx.h"
#include "shaders.h"
#include "device.h"
#include "../Utils/LZ4/lz4.h"

namespace
{
#ifdef _DEBUG
#include "shaderData-Debug.inl"
#else
#include "shaderData-Release.inl"
#endif

	// static std::vector<CComPtr<ID3D11ComputeShader>> s_shaders;
}

HRESULT DirectCompute::createComputeShaders( std::vector<CComPtr<ID3D11ComputeShader>>& shaders )
{
	constexpr size_t countBinaries = s_shaderOffsets.size() - 1;
	const size_t cbDecompressedLength = s_shaderOffsets[ countBinaries ];
	constexpr size_t countShaders = s_shaderBlobs32.size();

	std::vector<uint8_t> dxbc;
	try
	{
		shaders.resize( countShaders );
		dxbc.resize( cbDecompressedLength );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	const int lz4Status = LZ4_decompress_safe( (const char*)s_compressedShaders.data(), (char*)dxbc.data(), (int)s_compressedShaders.size(), (int)cbDecompressedLength );
	if( lz4Status != (int)cbDecompressedLength )
	{
		logError( u8"LZ4_decompress_safe failed with status %i", lz4Status );
		return PLA_E_CABAPI_FAILURE;
	}
	ID3D11Device* const dev = device();

	const auto& blobs = gpuInfo().wave64() ? s_shaderBlobs64 : s_shaderBlobs32;

	for( size_t i = 0; i < countShaders; i++ )
	{
		const size_t idxBinary = blobs[ i ];
		const uint32_t offThis = s_shaderOffsets[ idxBinary ];
		const uint8_t* rsi = &dxbc[ offThis ];
		const size_t len = s_shaderOffsets[ idxBinary + 1 ] - offThis;
		const HRESULT hr = dev->CreateComputeShader( rsi, len, nullptr, &shaders[ i ] );
		if( SUCCEEDED( hr ) )
			continue;

		const uint64_t binaryBit = ( 1ull << idxBinary );
		if( 0 != ( binaryBit & fp64ShadersBitmap ) )
			continue;	// This shader uses FP64 math, the support for that is optional. When not supported, CreateComputeShader method is expected to fail.
		// TODO [low]: ideally, query for the support when creating the device, and don't even try creating these compute shaders
		return hr;
	}

	return S_OK;
}