#include "stdafx.h"
#include "mlStartup.h"
#include "../D3D/startup.h"
#include "LookupTables.h"

namespace
{
	static DirectCompute::LookupTables s_tables;
}

namespace DirectCompute
{
	const LookupTables& lookupTables = s_tables;

	HRESULT mlStartup()
	{
		CHECK( d3dStartup() );
		CHECK( s_tables.create() );
		return S_OK;
	}

	void mlShutdown()
	{
		s_tables.clear();
		d3dShutdown();
	}
}