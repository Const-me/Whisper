#include "stdafx.h"
#include "tracing.h"
#include "../../source/ggml.h"
#include "../../ML/Tensor.h"

namespace Tracing
{
#if SAVE_DEBUG_TRACE
	std::unique_ptr<iTraceWriter> s_writer;

	static BOOL __stdcall consoleHandler( DWORD dwCtrlType )
	{
		if( dwCtrlType == CTRL_C_EVENT )
			s_writer = nullptr;

		// Return TRUE if handled this message, further handler functions won't be called.
		// Return FALSE to pass this message to further handlers until default handler calls ExitProcess().
		return FALSE;
	}

	void traceCreate( LPCTSTR path )
	{
		s_writer = iTraceWriter::create( path );
		SetConsoleCtrlHandler( &consoleHandler, TRUE );
	}

	void traceClose()
	{
		s_writer = nullptr;
	}

	iTraceWriter* getWriter()
	{
		return s_writer.get();
	}

	using Pair = std::pair<ItemName, ggml_tensor>;
	static std::vector<Pair> delayed;

	void delayTensor( const ItemName& name, const ggml_tensor* tensor )
	{
		delayed.emplace_back( name, *tensor );
	}

	HRESULT writeDelayedTensors()
	{
		if( delayed.empty() )
			return S_FALSE;
		iTraceWriter* w = getWriter();
		if( nullptr == w )
		{
			delayed.clear();
			return S_FALSE;
		}
		for( const Pair& p : delayed )
			w->tensor( p.first, p.second );
		delayed.clear();
		return S_OK;
	}
#elif DBG_TEST_NAN
	HRESULT tensor( const ItemName& name, const DirectCompute::Tensor& tensor )
	{
		const bool found = scanTensorForNaN( tensor, tensor.countElements() );
		if( !found )
			return S_FALSE;
		__debugbreak();
		return S_FALSE;
	}
#endif
}