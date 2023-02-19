#include "stdafx.h"
#include "mlStartup.h"
#include "../D3D/startup.h"
#include "LookupTables.h"
#include "../D3D/MappedResource.h"
#include "mlUtils.h"
#include "../D3D/shaders.h"

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

	void zeroMemory( ID3D11UnorderedAccessView* uav, uint32_t length, bool fillWithNaN )
	{
		__m128i cbData = _mm_cvtsi32_si128( (int)length );
		cbData = _mm_insert_epi32( cbData, fillWithNaN ? 1 : 0, 1 );
		ID3D11Buffer* cb = updateSmallCb( cbData );

		ID3D11DeviceContext* ctx = context();
		ctx->CSSetUnorderedAccessViews( 0, 1, &uav, nullptr );
		csSetCB( cb );

		constexpr uint32_t THREADS = 512;
		constexpr uint32_t ITERATIONS = 128;
		constexpr uint32_t elementsPerGroup = THREADS * ITERATIONS;
		const uint32_t countGroups = ( length + elementsPerGroup - 1 ) / elementsPerGroup;
		bindShader( eComputeShader::zeroMemory );
		ctx->Dispatch( countGroups, 1, 1 );
	}
}