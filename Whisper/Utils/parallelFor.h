#pragma once

namespace Whisper
{
	// A callback to offload to the thread pool
	using pfnParallelForCallback = HRESULT( * )( int ith, void* ctx ) noexcept;

	// A simple parallel for implementation; Windows includes a decent thread pool since Vista (2006)
	HRESULT parallelFor( pfnParallelForCallback pfn, int threadsCount, void* ctx );

	// Use this version when you wanna use the thread pool repeatedly, for the same work.
	// This class caches native work handle, saving a couple of WinAPI calls.
	class alignas( 64 ) ThreadPoolWork
	{
		PTP_WORK work = nullptr;

		// We want these volatile fields in another cache line from the rest of the data of this class.
		// threadIndex field is concurrently modified by different CPU cores, and these cache coherency protocols are slow.
		// OTOH, work and callback fields of this class only change when created / destroyed, that cache line is shared by CPU cores without any performance penalty.
		alignas( 64 ) volatile long threadIndex = 0;
		volatile HRESULT status = E_UNEXPECTED;

		static void __stdcall callbackStatic( PTP_CALLBACK_INSTANCE Instance, PVOID pv, PTP_WORK Work );

	protected:
		virtual HRESULT threadPoolCallback( int ith ) noexcept = 0;

	public:
		ThreadPoolWork() = default;
		ThreadPoolWork( const ThreadPoolWork& ) = delete;

		~ThreadPoolWork();

		HRESULT create();

		HRESULT parallelFor( int threadsCount ) noexcept;
	};
}