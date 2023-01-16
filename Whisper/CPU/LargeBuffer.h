#pragma once

namespace CpuCompute
{
	// A large memory buffer allocated with VirtualAlloc kernel API, bypassing the heap.
	class LargeBuffer
	{
		void* pv = nullptr;
	public:
		LargeBuffer() = default;
		LargeBuffer( const LargeBuffer& ) = delete;
		LargeBuffer( LargeBuffer&& that ) noexcept
		{
			pv = that.pv;
			that.pv = nullptr;
		}
		~LargeBuffer()
		{
			deallocate();
		}
		void operator=( LargeBuffer&& that ) noexcept
		{
			std::swap( pv, that.pv );
		}
		void operator=( const LargeBuffer& that ) = delete;

		// Allocate buffer with specified count of bytes, and read+write memory protection
		// The OS kernel guarantees zero-initialization of that memory.
		HRESULT allocate( size_t cb );

		// Change memory protection of the buffer to read only
		HRESULT setReadOnly( size_t cb );

		// Unless the pointer is nullptr, deallocate the buffer
		void deallocate();

		// Pointer to the start of the buffer, aligned by memory page = 4 kilobytes
		uint8_t* pointer() const
		{
			assert( nullptr != pv );
			return (uint8_t*)pv;
		}
	};
}