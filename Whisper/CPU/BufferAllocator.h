#pragma once
#include "LargeBuffer.h"
#include "Tensor.h"

namespace CpuCompute
{
#ifdef NDEBUG
	inline void dbgMarkUninitializedMemory( void* pv, size_t cb ) { }
	inline void dbgMarkFreedMemory( void* pv, size_t cb ) { }
#else
	void dbgMarkUninitializedMemory( void* pv, size_t cb );
	void dbgMarkFreedMemory( void* pv, size_t cb );
#endif

	// An implementation of arena allocator which slices pieces of a large buffer allocated in advance
	class BufferAllocator : public iArenaAllocator
	{
		LargeBuffer buffer;
		size_t head = 0;
		size_t size = 0;

		void resetArena() noexcept override final
		{
			head = 0;
			dbgMarkFreedMemory( buffer.pointer(), size );
		}

		void* allocate( size_t cb, size_t align ) noexcept override final;

	public:
		BufferAllocator() = default;
		BufferAllocator( const BufferAllocator& ) = delete;
		~BufferAllocator() = default;

		// Allocate a large buffer with the specified count of bytes
		HRESULT create( size_t cb );
	};

	// An implementation of arena allocator which allocates a large chunk of virtual memory, and maps new physical pages into that memory region as needed.
	class VirtualAllocator : public iArenaAllocator
	{
		uint8_t* pointer = nullptr;
		size_t head = 0;
		size_t sizeAllocated = 0;
		size_t sizeVirtual = 0;

		void resetArena() noexcept override final
		{
			head = 0;
			dbgMarkFreedMemory( pointer, sizeAllocated );
		}

		void* allocate( size_t cb, size_t align ) noexcept override final;

	public:

		VirtualAllocator() = default;
		VirtualAllocator( const VirtualAllocator& ) = delete;
		~VirtualAllocator();

		// Reserve virtual memory space for the specified count of bytes in the arena, but don't allocate any pages
		HRESULT create( size_t cb );
	};
}