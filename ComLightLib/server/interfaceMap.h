#pragma once
#include "../utils/typeTraits.hpp"

// Unlike ATL, the interface map is optional for ComLight.
// If you won't declare a map, the object will support 2 interfaces: IUnknown, and whatever template argument was passed to ObjectRoot class.
#define BEGIN_COM_MAP()                                      \
protected:                                                   \
bool implQueryInterface( REFIID iid, void** ppvObject ) {

#define END_COM_MAP() return false; }

namespace ComLight
{
	namespace details
	{
		template<typename I, typename C>
		inline bool tryReturnInterface( REFIID iid, C* pThis, void** ppvResult )
		{
			static_assert( pointersAssignable<IUnknown, I>(), "Trying to implement an interface that doesn't derive from IUnknown" );
			static_assert( pointersAssignable<I, C>(), "Declared support for an interface, but the class doesn't implement it" );
			if( I::iid() != iid )
				return false;
			I* const result = pThis;
			result->AddRef();
			*ppvResult = result;
			return true;
		}
	}
}

#define COM_INTERFACE_ENTRY( I ) if( ComLight::details::tryReturnInterface<I>( iid, this, ppvObject ) ) return true;