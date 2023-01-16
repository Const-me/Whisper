#include "stdafx.h"
#include "CpuProfiler.h"

namespace
{
	using namespace Whisper;

	inline int64_t qpcNow()
	{
		int64_t res;
		QueryPerformanceCounter( (LARGE_INTEGER*)&res );
		return res;
	}

	class CpuTimescale
	{
		uint64_t frequency = 0;
		const int64_t tscStart;
		const int64_t qpcStart;

		uint64_t computeTscFrequency();

	public:

		CpuTimescale() :
			tscStart( tscNow() ),
			qpcStart( qpcNow() )
		{ }

		inline uint64_t computeTicks( uint64_t tsc )
		{
			uint64_t freq = frequency;
			if( freq == 0 )
				freq = computeTscFrequency();

			return makeTime( tsc, freq );
		}
	};

	uint64_t __declspec( noinline ) CpuTimescale::computeTscFrequency()
	{
		int64_t tsc = tscNow();
		int64_t qpc = qpcNow();
		tsc -= tscStart;
		qpc -= qpcStart;

		uint64_t qpcFreq;
		QueryPerformanceFrequency( (LARGE_INTEGER*)&qpcFreq );

		// Seconds = qpc / qpcFreq
		// ticks per second = tsc / seconds = tsc * qpcFreq / qpc
		uint64_t res = ( (uint64_t)tsc * qpcFreq + ( (uint64_t)qpc / 2 ) - 1 ) / (uint64_t)qpc;
		frequency = res;
		const double GHz = (double)(int64_t)res * 1.0E-9;
		logDebug( u8"Computed CPU base frequency: %g GHz", GHz );
		return res;
	}

	static CpuTimescale timescale;
}

uint64_t Whisper::ticksFromTsc( uint64_t tscDiff )
{
	return timescale.computeTicks( tscDiff );
}