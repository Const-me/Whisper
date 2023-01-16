#pragma once
#include "RefCounter.hpp"
#include "../comLightCommon.h"
#include "../utils/typeTraits.hpp"

namespace ComLight
{
	// Base class of objects, implements reference counting, also a few lifetime methods.
	// The template argument is the interface you want clients to get when they ask for IID_IUnknown. By convention, that pointer defines object's identity.
	template<class I>
	class ObjectRoot : public RefCounter, public I
	{
	protected:

		inline HRESULT internalFinalConstruct()
		{
			return S_FALSE;
		}

		inline HRESULT FinalConstruct()
		{
			return S_FALSE;
		}

		inline void FinalRelease() { }

		IUnknown* getUnknown()
		{
			static_assert( details::pointersAssignable<IUnknown, I>(), "The interface doesn't derive from IUnknown" );
			return static_cast<I*>( this );
		}

		bool queryExtraInterfaces( REFIID riid, void **ppvObject ) const
		{
			return false;
		}

		// Implement query interface with 2 entries, IUnknown and I.
		bool implQueryInterface( REFIID riid, void** ppvObject )
		{
			if( riid == I::iid() || riid == IUnknown::iid() )
			{
				I* const result = this;
				result->AddRef();
				*ppvObject = result;
				return true;
			}
			return false;
		}
	};
}