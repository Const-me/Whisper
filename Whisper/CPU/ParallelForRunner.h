#pragma once
#include "LargeBuffer.h"

namespace CpuCompute
{
	// Callback interface for the parallel `for`
	__interface iComputeRange
	{
		// The implementation calls this method on multiple thread pool threads in parallel, and aggregates status codes.
		HRESULT __stdcall compute( size_t begin, size_t end ) const;
	};

	// Similar to ThreadPoolWork in parallelFor.h, optimized to be used as a direct replacement of OpenMP pool.
	class alignas( 64 ) ParallelForRunner
	{
	public:
		ParallelForRunner( int threads );
		~ParallelForRunner();

		HRESULT setThreadsCount( int threads );

		HRESULT parallelFor( iComputeRange& compute, size_t length, size_t minBatch = 1 );

		// Allocate a temporary buffer for the calling thread.
		// The pointer is guaranteed to be aligned by page size = 4kb
		void* threadLocalBuffer( size_t cb );

	private:

		int maxThreads;
		PTP_WORK work = nullptr;
		iComputeRange* computeRange = nullptr;
		size_t countItems = 0;
		size_t countThreads = 0;

		// Aligning by cache lines.
		// Avoiding cache line sharing between CPU cores improves performance, despite wasting a few bytes of memory.
		struct alignas( 64 ) ThreadBuffer
		{
			LargeBuffer memory;
			size_t cb = 0;
		};
		std::vector<ThreadBuffer> threadBuffers;

		alignas( 64 ) volatile long threadIndex = 0;
		volatile HRESULT status = S_OK;

		void runBatch( size_t ith ) noexcept;

		static void __stdcall workCallbackStatic( PTP_CALLBACK_INSTANCE Instance, void* pv, PTP_WORK Work ) noexcept;
	};
}