#pragma once
#include <vector>

namespace Whisper
{
	struct AudioBuffer
	{
		std::vector<float> mono;
		std::vector<float> stereo;

		void appendMono( const float* rsi, size_t countFloats );
		void appendDownmixedStereo( const float* rsi, size_t countFloats );
		void appendStereo( const float* rsi, size_t countFloats );

		using pfnAppendSamples = void( AudioBuffer::* )( const float* rsi, size_t countFloats );

		inline static pfnAppendSamples appendSamplesFunc( bool sourceMono, bool wantStereo )
		{
			if( sourceMono )
				return &AudioBuffer::appendMono;
			else if( !wantStereo )
				return &AudioBuffer::appendDownmixedStereo;
			else
				return &AudioBuffer::appendStereo;
		}

		void clear()
		{
			mono.clear();
			stereo.clear();
		}

		void swap( AudioBuffer& that )
		{
			mono.swap( that.mono );
			stereo.swap( that.stereo );
		}

		void resize( size_t len )
		{
			assert( len <= mono.size() );
			mono.resize( len );
			if( !stereo.empty() )
				stereo.resize( len * 2 );
		}
	};
}