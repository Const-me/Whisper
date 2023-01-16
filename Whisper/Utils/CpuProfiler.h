#pragma once

namespace Whisper
{
	// Get current time in CPU clock
	// More specifically, each CPU core has a timestamp counter which runs at CPU's base frequency, regardless on the frequency scaling of that core.
	inline int64_t tscNow()
	{
		return __rdtsc();
	}

	// Scale the time interval from CPU time stamp counter clock into 100-nanosecond ticks, rounding to nearest
	uint64_t ticksFromTsc( uint64_t tscDiff );

	class CpuProfiler
	{
		const int64_t started = tscNow();

	public:

		uint64_t elapsed() const
		{
			return ticksFromTsc( (uint64_t)( tscNow() - started ) );
		}
	};
}