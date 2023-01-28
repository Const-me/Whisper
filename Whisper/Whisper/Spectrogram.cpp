#include "stdafx.h"
#include "Spectrogram.h"
#include <memory>
#define _USE_MATH_DEFINES
#include <math.h>
#include "../Utils/parallelFor.h"
#include "../API/iMediaFoundation.cl.h"
#include "../ML/testUtils.h"
#include "melSpectrogram.h"
using namespace Whisper;

class alignas( 64 ) Spectrogram::MelContext
{
	const float* const samples;
	const size_t countSamples;
	Spectrogram& result;
	const int n_threads;
	SpectrogramContext context;

public:

	MelContext( const float* rsi, size_t len, const Filters& f, Spectrogram& rdi, int countThreads ) :
		samples( rsi ), countSamples( len ), result( rdi ), n_threads( countThreads ),
		context( f )
	{ }

	void run( int ith );

	static HRESULT workCallback( int ith, void* ctx ) noexcept;
};

void Spectrogram::MelContext::run( int ith )
{
	std::array<float, N_MEL> arr;
	for( uint32_t i = ith; i < result.length; i += n_threads )
	{
		const int offset = i * FFT_STEP;
		const float* rsi = samples + offset;
		context.fft( arr, rsi, countSamples - offset );

		for( size_t j = 0; j < N_MEL; j++ )
			result.data[ j * result.length + i ] = arr[ j ];
	}
}

HRESULT Spectrogram::MelContext::workCallback( int ith, void* ctx ) noexcept
{
	std::vector<Spectrogram::MelContext>& contexts = *( std::vector<Spectrogram::MelContext>* )ctx;
	try
	{
		contexts.at( ith ).run( ith );
		return S_OK;
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}
	catch( const std::exception& )
	{
		return E_FAIL;
	}
}

HRESULT Spectrogram::pcmToMel( const iAudioBuffer* buffer, const Filters& filters, int threads )
{
	if( nullptr == buffer )
		return E_POINTER;
	const uint32_t countSamples = buffer->countSamples();
	if( 0 == countSamples )
		return OLE_E_BLANK;
	const float* const samples = buffer->getPcmMono();

	length = ( countSamples ) / FFT_STEP;
	data.resize( N_MEL * length );

	if( threads < 2 )
	{
		MelContext ctx{ samples, countSamples, filters, *this, 1 };
		ctx.run( 0 );
	}
	else
	{
		std::vector<MelContext> contexts;
		contexts.reserve( threads );
		for( int i = 0; i < threads; i++ )
			contexts.emplace_back( MelContext{ samples, countSamples, filters, *this, (int)threads } );
		CHECK( parallelFor( &MelContext::workCallback, threads, &contexts ) );
	}

	// clamping and normalization
	double mmax = -1e20;
	for( double f : data )
		mmax = std::max( mmax, f );
	//printf("%s: max = %f\n", __func__, mmax);

	mmax -= 8.0;

	for( float& f : data )
	{
		if( f < mmax )
			f = (float)mmax;
		f = (float)( ( f + 4.0 ) / 4.0 );
	}
	// DirectCompute::dbgWriteBinaryFile( LR"(C:\Temp\2remove\ML\mel-my.bin)", data.data(), data.size() * 4 );
	const float* const pcmStereo = buffer->getPcmStereo();
	if( nullptr != pcmStereo )
	{
		try
		{
			stereo.resize( countSamples );
		}
		catch( const std::bad_alloc& )
		{
			return E_OUTOFMEMORY;
		}
		memcpy( stereo.data(), pcmStereo, countSamples * 8 );
	}
	else
		stereo.clear();

	return S_OK;
}

void Whisper::computeSignalEnergy( std::vector<float>& result, const iAudioBuffer* buffer, int n_samples_per_half_window )
{
	const size_t countSamples = buffer->countSamples();
	const float* const samples = buffer->getPcmMono();

	const int hw = n_samples_per_half_window;
	result.resize( countSamples );

	for( size_t i = 0; i < countSamples; i++ )
	{
		float sum = 0;
		for( int j = -hw; j <= hw; j++ )
			if( i + j >= 0 && i + j < countSamples )
				sum += fabsf( samples[ i + j ] );
		result[ i ] = sum / ( 2 * hw + 1 );
	}
}

HRESULT Spectrogram::copyStereoPcm( size_t offset, size_t length, std::vector<StereoSample>& buffer ) const
{
	if( stereo.empty() )
		return OLE_E_BLANK;

	length *= FFT_STEP;
	offset *= FFT_STEP;
	if( offset >= stereo.size() )
		return E_BOUNDS;

	try
	{
		buffer.resize( length );
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	const size_t lengthToCopy = std::min( length, stereo.size() - offset );
	memcpy( buffer.data(), &stereo[ offset ], lengthToCopy * 8 );
	if( lengthToCopy == length )
		return S_OK;

	memset( &buffer[ lengthToCopy ], 0, ( buffer.size() - lengthToCopy ) * 8 );
	return S_OK;
}