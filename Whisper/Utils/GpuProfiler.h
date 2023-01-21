#pragma once
#include "../D3D/device.h"
#include "ProfileCollection.h"
#include "DelayExecution.h"

namespace DirectCompute
{
	enum struct eProfilerBlock : uint16_t
	{
		LoadModel = 0x1000,
		Run = 0x2000,
		Encode = 0x3000,
		EncodeLayer = 0x4000,
		Decode = 0x5000,
		DecodeStep = 0x6000,
		DecodeLayer = 0x7000,
	};

	enum struct eComputeShader : uint16_t;

	class GpuProfiler
	{
		DelayExecution delay;
		CComPtr<ID3D11Query> disjoint;

		enum struct eEvent
		{
			None = 0,
			BlockStart,
			BlockEnd,
			Shader
		};

		struct BlockState;
		static constexpr uint16_t EmptyShader = ~(uint16_t)0;

		// A circular buffer with in-flight queries which feeds timestamps into the iTimestampSink interface
		class Queue
		{
			static constexpr size_t queueLength = 32;

			// Ring buffer for individual measures
			struct Entry
			{
				CComPtr<ID3D11Query> query;
				BlockState* block;
				eEvent event;
				uint16_t shader;
#if PROFILER_COLLECT_TAGS
				uint16_t tag = 0;
#endif
				void join( GpuProfiler& owner );
			};

			GpuProfiler& owner;
			std::array<Entry, queueLength> queue;
			size_t nextEntry = 0;

		public:
			Queue( GpuProfiler& gp ) : owner( gp ) {}

			HRESULT create();

			// Begin a next query. Eventually, this will result in the BlockState.haveTimestamp callback
			void submit( BlockState* block, eEvent evt, uint16_t shader = EmptyShader, uint16_t tag = 0 );

			// Wait for all the pending queries, and call their callbacks
			void join();
		};
		Queue queries;

		struct sProfilerData;
		struct BlockState
		{
			int64_t timeStart = -1;
			sProfilerData* destBlock = nullptr;
			int64_t shaderStart = -1;
			uint16_t prevShader = EmptyShader;
			uint16_t prevShaderTag = 0;
			BlockState* parentBlock = nullptr;
			void haveTimestamp( eEvent evt, uint16_t cs, uint16_t tag, uint64_t time, GpuProfiler& profiler );
		private:
			void completePrevShader( uint64_t time, GpuProfiler& profiler );
		};
		CAtlMap<eProfilerBlock, BlockState> blockStates;
		std::vector<BlockState*> stack;

		struct sProfilerData
		{
			// Count of accumulated measures
			size_t callsPending;
			// Total time spent running all instances of that measure, expressed in GPU ticks
			uint64_t timePending;

			Whisper::ProfileCollection::Measure* dest;

			inline void makeTime( uint64_t freq );
			inline void addPending( int64_t time );
			inline void reset();
			inline void dropPending();

			sProfilerData()
			{
				reset();
			}
		};

		CAtlMap<uint16_t, sProfilerData> results;
#if PROFILER_COLLECT_TAGS
		CAtlMap<uint32_t, sProfilerData> resultsTagged;
#endif
		void resultsMakeTime( uint64_t freq );
		void resultsDropPending();
		void resultsReset();

		void blockStart( eProfilerBlock which );
		void blockEnd();

		Whisper::ProfileCollection& dest;
#if PROFILER_COLLECT_TAGS
		uint16_t m_nextTag = 0;
#endif
	public:

		GpuProfiler( Whisper::ProfileCollection& pc ) :
			dest( pc ), queries( *this ) { }

		HRESULT create( size_t maxDepth = 3 );

		class BlockRaii
		{
			GpuProfiler* profiler;

		public:
			BlockRaii( GpuProfiler& owner, eProfilerBlock which )
			{
				owner.blockStart( which );
				profiler = &owner;
			}
			~BlockRaii()
			{
				if( nullptr != profiler )
				{
					profiler->blockEnd();
					profiler = nullptr;
				}
			}
			BlockRaii( BlockRaii&& that ) noexcept :
				profiler( that.profiler )
			{
				that.profiler = nullptr;
			}
			BlockRaii( const BlockRaii& ) = delete;
			void operator=( const BlockRaii& ) = delete;
			void operator=( BlockRaii&& ) = delete;
		};

		BlockRaii block( eProfilerBlock which )
		{
			return BlockRaii{ *this, which };
		}

		void computeShader( eComputeShader cs );

		bool profileShaders = false;
		// bool profileShaders = true;

		decltype( auto ) cpuBlock( Whisper::eCpuBlock block )
		{
			return dest.cpuBlock( block );
		}
		Whisper::ProfileCollection& profiler() { return dest; }

		// Set tag string for the next compute shader
		// The string should be readonly: for performance reason the implementation doesn’t copy nor compare any strings, it only keeps the pointer
#if PROFILER_COLLECT_TAGS
		uint16_t setNextTag( const char* name );
#else
		inline uint16_t setNextTag( const char* name ) { return 0; }
#endif

		void setNextTag( uint16_t tag )
		{
#if PROFILER_COLLECT_TAGS
			m_nextTag = tag;
#endif
		}
	};
}