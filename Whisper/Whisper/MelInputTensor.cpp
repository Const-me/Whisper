#include "stdafx.h"
#include "MelInputTensor.h"
#include "../D3D/MappedResource.h"
#include "../D3D/createBuffer.h"
#include <mfapi.h>	// MFCopyImage
using namespace DirectCompute;

HRESULT MelInputTensor::create( Whisper::iSpectrogram& spectrogram, const sEncodeParams& encParams )
{
	// Ported from the initial portion of whisper_encode() function
	const size_t ne0 = encParams.n_ctx * 2;
	const size_t ne1 = encParams.n_mels;
	const size_t totalElts = ne0 * ne1;
	const size_t totalBytes = totalElts * 4;

	if( capacity < (uint32_t)totalElts )
	{
		// The old buffer is too small: drop the old one, and create a larger buffer with SRV
		buffer = nullptr;
		TensorGpuViews::clear();

		CHECK( createBuffer( eBufferUse::Dynamic, totalBytes, &buffer, nullptr, nullptr ) );
		CHECK( TensorGpuViews::create( buffer, DXGI_FORMAT_R32_FLOAT, totalElts, false ) );

		capacity = (uint32_t)totalElts;
	}

	// Upload data to VRAM using D3D11_MAP_WRITE_DISCARD, that's why we made a dynamic buffer
	{
		// Ported from whisper_encode() function
		MappedResource mapped;
		CHECK( mapped.map( buffer, false ) );
		float* const dst = (float*)mapped.data();
		memset( dst, 0, totalBytes );

		const size_t n_len = spectrogram.getLength();
		const size_t i0 = std::min( (size_t)encParams.mel_offset, n_len );
		const size_t i1 = std::min( (size_t)encParams.mel_offset + 2 * encParams.n_ctx, n_len );

		// Whisper::MelBufferRaii sourceBuffer{ spectrogram, i0, i1 - i0 };
		constexpr DWORD n_mel = Whisper::N_MEL;
		const size_t rowBytes = ( i1 - i0 ) * 4;
		/*
		for( size_t j = 0; j < n_mel; j++ )
		{
			float* rdi = dst + j * 2 * encParams.n_ctx;
			const float* rsi = sourceBuffer[ j ];
			memcpy( rdi, rsi, rowBytes );
		} */

		Whisper::MelBufferRaii sourceBuffer;
		CHECK( sourceBuffer.make( spectrogram, i0, i1 - i0 ) );
		CHECK( MFCopyImage(
			(BYTE*)dst, (LONG)( 2 * encParams.n_ctx * sizeof( float ) ),
			sourceBuffer.bytePtr(), sourceBuffer.strideBytes(),
			(DWORD)rowBytes, n_mel ) );
	}

	// Shape the tensor
	ne = { 2 * encParams.n_ctx, encParams.n_mels, 1, 1 };
	TensorShape::setDenseStrides();
	return S_OK;
}