#include "stdafx.h"
#include "mlStartup.h"
#include "../D3D/startup.h"
#include "LookupTables.h"
#include "../D3D/MappedResource.h"
#include "mlUtils.h"

namespace
{
	static DirectCompute::LookupTables s_tables;
	static CComPtr<ID3D11Buffer> s_smallCb;
}

namespace DirectCompute
{
	const LookupTables& lookupTables = s_tables;

	HRESULT mlStartup( uint32_t flags )
	{
		CHECK( d3dStartup( flags ) );
		CHECK( s_tables.create() );
		{
			CD3D11_BUFFER_DESC desc{ 16, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE };
			CHECK( device()->CreateBuffer( &desc, nullptr, &s_smallCb ) );
		}
		return S_OK;
	}

	void mlShutdown()
	{
		s_smallCb = nullptr;
		s_tables.clear();
		d3dShutdown();
	}

	ID3D11Buffer* __vectorcall updateSmallCb( __m128i cbData )
	{
		ID3D11Buffer* cb = s_smallCb;
		if( nullptr != cb )
		{
			MappedResource mapped;
			check( mapped.map( cb, false ) );
			store16( mapped.data(), cbData );
		}
		throw OLE_E_BLANK;
	}
}