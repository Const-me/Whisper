#pragma once
#include "comLightCommon.h"
#include "client/CComPtr.hpp"
#include "utils/typeTraits.hpp"

namespace ComLight
{
	namespace details
	{
		template<typename T>
		inline constexpr void** castDoublePointerToVoid( T** pp )
		{
			static_assert( pointersAssignable<IUnknown, T>(), "IID_PPV_ARGS macro should be used with IUnknown interfaces" );
			return reinterpret_cast<void**>( pp );
		}
	}
}

#ifdef IID_PPV_ARGS
#undef IID_PPV_ARGS
#endif

#define IID_PPV_ARGS( pp ) decltype( **pp )::iid, ::ComLight::details::castDoublePointerToVoid( pp )