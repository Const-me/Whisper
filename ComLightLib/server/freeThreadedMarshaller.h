#pragma once
#ifdef _MSC_VER
#include "../comLightCommon.h"

namespace ComLight
{
	namespace details
	{
		HRESULT createFreeThreadedMarshaller( IUnknown* pUnkOuter, IUnknown** ppUnkMarshal );
		bool queryMarshallerInterface( REFIID riid, void **ppvObject, IUnknown* marshaller );
	}
}

#define DECLARE_FREE_THREADED_MARSHALLER()                                                              \
private:                                                                                                \
ComLight::CComPtr<ComLight::IUnknown> m_freeThreadedMarshaller;                                         \
protected:                                                                                              \
HRESULT internalFinalConstruct()                                                                        \
{                                                                                                       \
	return ComLight::details::createFreeThreadedMarshaller( getUnknown(), &m_freeThreadedMarshaller );  \
}                                                                                                       \
bool queryExtraInterfaces( REFIID riid, void **ppvObject ) const                                        \
{                                                                                                       \
	return ComLight::details::queryMarshallerInterface( riid, ppvObject, m_freeThreadedMarshaller );    \
}

#else
#define DECLARE_FREE_THREADED_MARSHALLER()
#endif