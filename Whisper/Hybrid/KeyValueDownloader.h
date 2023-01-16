#pragma once
#include "../Whisper/sModelParams.h"
#include "../Whisper/KeyValueBuffers.h"
#include "../D3D/MappedResource.h"
#include "../CPU/Tensor.h"

class KeyValueDownloader
{
	CComPtr<ID3D11Buffer> keys, values;
	uint32_t length = 0;

	using E = uint16_t;
	static constexpr DirectCompute::eDataType dataType = DirectCompute::eDataType::FP16;

public:
	// Create the staging resources to download kvCross tensors produced by the GPGPU encoder
	HRESULT create( const Whisper::sModelParams& mp );

	// Download these two tensors from VRAM to the staging buffers in system RAM
	HRESULT download( const DirectCompute::KeyValueBuffers& source );

	class ReadMap
	{
		const uint32_t length;
		DirectCompute::MappedResource mappedKeys, mappedValues;

	public:
		ReadMap( KeyValueDownloader& owner );
		~ReadMap() = default;
		ReadMap( const ReadMap& ) = delete;

		// A slice of model.memory_k tensor
		CpuCompute::Tensor keysView( uint32_t len, uint32_t off ) const
		{
			if( len + off <= length )
			{
				E* rsi = (E*)mappedKeys.data();
				rsi += off;
				return CpuCompute::Tensor::fromData( rsi, dataType, len );
			}
			throw E_BOUNDS;
		}

		// A slice of model.memory_v tensor
		CpuCompute::Tensor valuesView( uint32_t len, uint32_t off ) const
		{
			if( len + off <= length )
			{
				E* rsi = (E*)mappedValues.data();
				rsi += off;
				return CpuCompute::Tensor::fromData( rsi, dataType, len );
			}
			throw E_BOUNDS;
		}
	};

	// Map both staging buffers, return RAII object which unmaps when destroyed,
	// which can supply the data in the shape of CpuCompute::Tensor vector
	decltype( auto ) map()
	{
		return ReadMap( *this );
	}
};