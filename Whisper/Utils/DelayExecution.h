#pragma once
#include <atlbase.h>

// Utility class implementing a high-resolution Sleep() function
class DelayExecution
{
	using pfnDelay = void( * )( const DelayExecution& de );
	pfnDelay pfn = nullptr;
	CHandle timer;

	static void sleepOnTheTimer( const DelayExecution& delay );
	static void spinWait( const DelayExecution& );
	static void sleep( const DelayExecution& );

public:
	DelayExecution();
	DelayExecution( const DelayExecution& ) = delete;
	~DelayExecution() = default;

	void delay() const
	{
		pfn( *this );
	}
};