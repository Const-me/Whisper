#pragma once
#include <deque>
#include "../MF/PcmReader.h"
#include "melSpectrogram.h"
#include "iSpectrogram.h"
#include <atlbase.h>
#include "../Utils/parallelFor.h"
#include "../Utils/ProfileCollection.h"
#include "../API/iMediaFoundation.cl.h"

namespace Whisper
{
	// Base class for both single- and multi-threaded MEL streamers
	// Used by iContext.runStreamed method
	class MelStreamer : public iSpectrogram
	{
	protected:
		PcmReader reader;
		std::deque<PcmMonoChunk> queuePcmMono;
		using MelChunk = std::array<float, N_MEL>;
		std::deque<MelChunk> queueMel;
		size_t streamStartOffset = 0;
		std::vector<float> tempPcm;
		std::vector<float> outputMel;
		SpectrogramContext melContext;
		bool readerEof = false;
		ProfileCollection& profiler;
		std::deque<PcmStereoChunk> queuePcmStereo;

		// If the streamStartOffset value is less than the argument,
		// remove ( off - streamStartOffset ) chunks from the start of all 3 queues, and advance streamStartOffset to the `off` argument
		void dropOldChunks( size_t off );

		// Ensure PCM queues have enough chunks to generate specified count of MEL chunks
		// At the end of the stream, the method delivers less chunks then requested and returns S_FALSE
		HRESULT ensurePcmChunks( size_t len );

		// Copy mono PCM chunks from the queue (starting at the specified element index) into the continuous tempPcm vector
		// Returns count of chunks copied there.
		size_t serializePcm( size_t startOffset );

		size_t lastBufferEnd = ~(size_t)0;
		float lastBufferMax = 0.0f;
		void makeTransposedBuffer( size_t off, size_t len );

		size_t getLength() const noexcept override final { return reader.getLength(); }

		HRESULT copyStereoPcm( size_t offset, size_t length, std::vector<StereoSample>& buffer ) const override final;

	public:
		MelStreamer( const Filters& filters, ProfileCollection& profiler, const iAudioReader* reader );
	};

	// Single-threaded MEL streamer: runs these FFTs on-demand, from within makeBuffer() method
	// Used by iContext.runStreamed method when cpuThreads parameter is less than 2
	class MelStreamerSimple : public MelStreamer
	{
		HRESULT makeBuffer( size_t offset, size_t length, const float** buffer, size_t& stride ) noexcept override final;

	public:
		MelStreamerSimple( const Filters& filters, ProfileCollection& profiler, const iAudioReader* reader ) :
			MelStreamer( filters, profiler, reader ) { }
	};

	// Multi threaded MEL streamers: runs FFT on a background thread ahead of time
	// The background thread tries to keep the queueMel full, this way the makeBuffer() method has very little to do
	// makeBuffer() only transposes the data, and does clamping + normalization, both steps are pretty fast
	// Used by iContext.runStreamed method when cpuThreads parameter is 2 or more
	class MelStreamerThread : public MelStreamer,
		ThreadPoolWork
	{
		HRESULT makeBuffer( size_t offset, size_t length, const float** buffer, size_t& stride ) noexcept override final;

		static DWORD __stdcall threadProcStatic( void* lpParameter );
		HRESULT run() noexcept;
		HRESULT threadMain();

		std::vector<MelChunk> pendingChunks;
		int fftChunks = 0;
		int fftThreads = 0;
		std::vector<SpectrogramContext> melContextsWorkers;
		CComAutoCriticalSection m_cs;
		CONDITION_VARIABLE wakeMain, wakeBackground;
		const int workerThreads;
		enum struct eThreadStatus : uint8_t
		{
			NotStarted = 0,
			Idle,
			Working,
			Completed,
			Failed
		};
		eThreadStatus threadStatus;
		bool shuttingDown = false;
		CHandle threadHandle;

		HRESULT threadPoolCallback( int ith ) noexcept override final;

	public:

		MelStreamerThread( const Filters& filters, ProfileCollection& profiler, const iAudioReader* reader, int countThreads );

		~MelStreamerThread();
	};
}