#include "stdafx.h"
#include "LookupTables.h"
#include "LookupTablesData.h"
#include <memory>
#include "mlUtils.h"
using namespace DirectCompute;

namespace
{
	HRESULT uploadLookupTable( const std::array<uint16_t, 0x10000>& rsi, CComPtr<ID3D11ShaderResourceView>& rdi )
	{
		rdi = nullptr;
		CComPtr<ID3D11Buffer> buffer;

		CD3D11_BUFFER_DESC desc{ 0x10000 * 2, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_IMMUTABLE };
		if( gpuInfo().cloneableModel() )
		{
			desc.Usage = D3D11_USAGE_DEFAULT;
			desc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
		}
		D3D11_SUBRESOURCE_DATA srd{ rsi.data(), 0, 0 };
		CHECK( device()->CreateBuffer( &desc, &srd, &buffer ) );

		CD3D11_SHADER_RESOURCE_VIEW_DESC viewDesc{ D3D11_SRV_DIMENSION_BUFFER, DXGI_FORMAT_R16_UINT, 0, 0x10000 };
		CHECK( device()->CreateShaderResourceView( buffer, &viewDesc, &rdi ) );

		return S_OK;
	}
}

HRESULT LookupTables::create()
{
	std::unique_ptr<LookupTablesData> data;
	try
	{
		data = std::make_unique<LookupTablesData>();
	}
	catch( const std::bad_alloc& )
	{
		return E_OUTOFMEMORY;
	}

	CHECK( uploadLookupTable( data->gelu, m_gelu ) );
	CHECK( uploadLookupTable( data->exponent, m_exponent ) );

	return S_OK;
}

HRESULT LookupTables::createClone( const LookupTables& source )
{
	CHECK( cloneResourceView( source.m_gelu, &m_gelu ) );
	CHECK( cloneResourceView( source.m_exponent, &m_exponent ) );
	return S_OK;
}

void LookupTables::clear()
{
	m_gelu = nullptr;
	m_exponent = nullptr;
}

__m128i LookupTables::getMemoryUsage() const
{
	__m128i v = resourceMemoryUsage( m_gelu );
	v = _mm_add_epi64( v, resourceMemoryUsage( m_exponent ) );
	return v;
}