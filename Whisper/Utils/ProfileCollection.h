#pragma once
#include <atlcoll.h>
#include "CpuProfiler.h"

namespace DirectCompute
{
	enum struct eComputeShader : uint16_t;
	enum struct eProfilerBlock : uint16_t;
}

namespace Whisper
{
	struct WhisperModel;

	enum struct eCpuBlock : uint8_t
	{
		LoadModel,
		RunComplete,
		Run,
		Callbacks,
		Spectrogram,
		Sample,
		VAD,
		Encode,
		Decode,
		DecodeStep,
		DecodeLayer,
	};

	class ProfileCollection
	{
	public:
		ProfileCollection( const WhisperModel& model );

		struct Measure
		{
			size_t count = 0;
			// 100-nanosecond ticks
			uint64_t totalTicks = 0;

			void reset()
			{
				count = 0;
				totalTicks = 0;
			}

			void print( const char* name ) const;

			void add( uint64_t val )
			{
				count++;
				totalTicks += val;
			}
		};

		Measure& measure( DirectCompute::eProfilerBlock which );
		Measure& measure( DirectCompute::eComputeShader which );
		Measure& measure( eCpuBlock which );
#if PROFILER_COLLECT_TAGS
		Measure& measure( DirectCompute::eComputeShader which, uint16_t tag );
#endif
		void print();

		void reset();

		class CpuRaii
		{
			Measure* dest;
			const int64_t tsc;

		public:
			CpuRaii( Measure& m ) : dest( &m ), tsc( tscNow() )
			{ }
			CpuRaii( const CpuRaii& ) = delete;
			CpuRaii( CpuRaii&& that ) noexcept :
				tsc( that.tsc )
			{
				dest = that.dest;
				that.dest = nullptr;
			}

			~CpuRaii()
			{
				if( nullptr != dest )
				{
					const int64_t elapsed = tscNow() - tsc;
					dest->add( ticksFromTsc( elapsed ) );
				}
			}
		};

		decltype( auto ) cpuBlock( eCpuBlock which )
		{
			return CpuRaii{ measure( which ) };
		}

		uint16_t makeTagId( const char* tag );

	private:
		CAtlMap<uint32_t, Measure> measures;
		CComAutoCriticalSection critSec;
#if PROFILER_COLLECT_TAGS
		CAtlMap<const char*, uint16_t> tagIDs;
		std::vector<const char*> tagNames;
		CAtlMap<uint32_t, Measure> taggedShaders;
		std::vector<uint32_t> taggedKeysTemp;
		struct TaggedTemp
		{
			uint64_t ticks;
			size_t count;
			const char* name;

			bool operator<( const TaggedTemp& that ) const
			{
				// Flipping the comparison to sort in descending order
				return ticks > that.ticks;
			}

			void print() const;
		};
		std::vector<TaggedTemp> taggedTimes;
#endif
		std::vector<uint32_t> keysTemp;
	};
}