#include "stdafx.h"
#include "shaders.h"
#include "startup.h"
#include "device.h"
#include <compressapi.h>
#pragma comment( lib, "Cabinet.lib" )

namespace
{
#ifdef _DEBUG
#include "shaderData-Debug.inl"
#else
#include "shaderData-Release.inl"
#endif

	constexpr DWORD compressionAlgorithm = COMPRESS_ALGORITHM_MSZIP;

	class Decompressor
	{
		DECOMPRESSOR_HANDLE handle = nullptr;

	public:

		HRESULT create()
		{
			if( CreateDecompressor( compressionAlgorithm, nullptr, &handle ) )
				return S_OK;
			return HRESULT_FROM_WIN32( GetLastError() );
		}

		HRESULT decompress( const uint8_t* src, size_t compressedLength, void* dest, size_t origLength ) const
		{
			if( Decompress( handle, src, compressedLength, dest, origLength, nullptr ) )
				return S_OK;
			return HRESULT_FROM_WIN32( GetLastError() );
		}

		~Decompressor()
		{
			if( nullptr != handle )
			{
				CloseDecompressor( handle );
				handle = nullptr;
			}
		}
	};

	static std::vector<CComPtr<ID3D11ComputeShader>> s_shaders;
}

HRESULT DirectCompute::createComputeShaders()
{
	constexpr size_t countBinaries = s_shaderOffsets.size() - 1;
	const size_t cbDecompressedLength = s_shaderOffsets[ countBinaries ];
	constexpr size_t countShaders = s_shaderBlobs32.size();

	std::vector<uint8_t> dxbc;
	try
	{
		s_shaders.resize( countShaders );
		dxbc.resize( cbDecompressedLength );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	Decompressor decomp;
	CHECK( decomp.create() );

	decomp.decompress( s_compressedShaders.data(), s_compressedShaders.size(), dxbc.data(), cbDecompressedLength );
	ID3D11Device* const dev = device();

	const auto& blobs = gpuInfo.wave64() ? s_shaderBlobs64 : s_shaderBlobs32;

	for( size_t i = 0; i < countShaders; i++ )
	{
		const size_t idxBinary = blobs[ i ];
		const uint32_t offThis = s_shaderOffsets[ idxBinary ];
		const uint8_t* rsi = &dxbc[ offThis ];
		const size_t len = s_shaderOffsets[ idxBinary + 1 ] - offThis;
		const HRESULT hr = dev->CreateComputeShader( rsi, len, nullptr, &s_shaders[ i ] );
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

void DirectCompute::destroyComputeShaders()
{
	s_shaders.clear();
}

void DirectCompute::bindShader( eComputeShader shader )
{
	context()->CSSetShader( s_shaders[ (uint16_t)shader ], nullptr, 0 );
}