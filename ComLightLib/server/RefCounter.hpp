#pragma once
#include <atomic>
#include <assert.h>
#include <limits.h>

namespace ComLight
{
	// Very base class of objects, implements reference counting.
	class RefCounter
	{
		std::atomic_uint referenceCounter;

	public:

		RefCounter() : referenceCounter( 0 ) { }

		inline virtual ~RefCounter() { }

		RefCounter( const RefCounter &that ) = delete;
		RefCounter( RefCounter &&that ) = delete;

	protected:

		uint32_t implAddRef()
		{
			return ++referenceCounter;
		}

		uint32_t implRelease()
		{
			// Might be a good idea to use locks, at least in debug builds. They're much slower than atomics, but with locks it's possible to detect when 2 threads call release at the same time, for object with counter = 1.
			// It's a memory management bug, but it would be nice if debug builds would handle that case gracefully.
			const uint32_t rc = --referenceCounter;
			assert( rc != UINT_MAX );
			return rc;
		}
	};
}