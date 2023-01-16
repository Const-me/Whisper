#include <stdafx.h>
#include "BufferAllocator.h"
#include <immintrin.h>
#include <ammintrin.h>
using namespace CpuCompute;

HRESULT BufferAllocator::create( size_t cb )
{
	CHECK( buffer.allocate( cb ) );
	head = 0;
	size = cb;
	dbgMarkUninitializedMemory( buffer.pointer(), cb );
	return S_OK;
}

namespace
{
	// Round up the integer by 32 bytes
	__forceinline size_t roundUpAlloc( size_t cb )
	{
		const size_t mask = 31;
		cb += mask;
		// We require AVX1+FMA3 support, might as well use BMI1
		return _andn_u64( mask, cb );
	}
}

void* BufferAllocator::allocate( size_t cb, size_t align ) noexcept
{
	assert( align <= 32 );
	cb = roundUpAlloc( cb );

	uint8_t* pointer = buffer.pointer();
	if( head + cb > size || nullptr == pointer )
	{
		logError( u8"BufferAllocator.allocate, not enough capacity" );
		return nullptr;
	}

	void* const res = pointer + head;
	head += cb;
	assert( head <= size );
	dbgMarkUninitializedMemory( res, cb );
	return res;
}

namespace
{
	// 2 MB of memory, we hope the OS kernel will then be smart enough to give us large pages.
	constexpr size_t virtualAllocGranularityExp2 = 21;

	constexpr size_t virtualAllocGranularityMask = ( ( (size_t)1 ) << virtualAllocGranularityExp2 ) - 1;

	// Round up the integer by 2 megabytes
	__forceinline size_t roundUpVirtualAlloc( size_t cb )
	{
		const size_t mask = virtualAllocGranularityMask;
		cb += mask;
		return _andn_u64( mask, cb );
	}
}

HRESULT VirtualAllocator::create( size_t cb )
{
	if( nullptr != pointer )
		return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );
	cb = roundUpVirtualAlloc( cb );
	pointer = (uint8_t*)VirtualAlloc( NULL, cb, MEM_RESERVE, PAGE_READWRITE );
	if( nullptr != pointer )
	{
		head = 0;
		sizeAllocated = 0;
		sizeVirtual = cb;
		return S_OK;
	}

	const HRESULT hr = getLastHr();
	logErrorHr( hr, u8"VirtualAlloc failed" );
	return hr;
}

void* VirtualAllocator::allocate( size_t cb, size_t align ) noexcept
{
	assert( align <= 32 );
	cb = roundUpAlloc( cb );

	const size_t newHead = head + cb;
	if( newHead <= sizeAllocated )
	{
		void* const res = pointer + head;
		head = newHead;
		dbgMarkUninitializedMemory( res, cb );
		return res;
	}

	if( newHead <= sizeVirtual )
	{
		uint8_t* const ptrCommit = pointer + sizeAllocated;
		const size_t cbCommit = roundUpVirtualAlloc( newHead ) - sizeAllocated;
		void* const res = VirtualAlloc( ptrCommit, cbCommit, MEM_COMMIT, PAGE_READWRITE );
		if( nullptr != res )
		{
			sizeAllocated += cbCommit;
			assert( sizeAllocated <= sizeVirtual );
			void* const res = pointer + head;
			head = newHead;
			dbgMarkUninitializedMemory( res, cb );
			return res;
		}

		const HRESULT hr = getLastHr();
		logErrorHr( hr, u8"VirtualAllocator.allocate, VirtualAlloc failed" );
		return nullptr;
	}

	logError( u8"VirtualAllocator.allocate, not enough arena capacity" );
	return nullptr;
}

VirtualAllocator::~VirtualAllocator()
{
	if( nullptr == pointer )
		return;

	if( VirtualFree( pointer, 0, MEM_RELEASE ) )
	{
		pointer = nullptr;
		return;
	}

	const HRESULT hr = getLastHr();
	logErrorHr( hr, u8"VirtualFree failed" );
}

#ifndef NDEBUG
// Reusing Microsoft's magic numbers: https://asawicki.info/news_1292_magic_numbers_in_visual_c
void CpuCompute::dbgMarkUninitializedMemory( void* pv, size_t cb )
{
	__stosb( (uint8_t*)pv, 0xCD, cb );
}
void CpuCompute::dbgMarkFreedMemory( void* pv, size_t cb )
{
	__stosd( (DWORD*)pv, 0xFEEEFEEEu, cb / 4 );
}
#endif