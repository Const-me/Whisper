#pragma once
#include "../D3D/device.h"

namespace DirectCompute
{
	class LookupTables
	{
		CComPtr<ID3D11ShaderResourceView> m_gelu, m_exponent;

	public:

		HRESULT create();
		HRESULT createClone( const LookupTables& source );
		void clear();
		ID3D11ShaderResourceView* gelu() const { return m_gelu; }
		ID3D11ShaderResourceView* exponent() const { return m_exponent; }

		__m128i getMemoryUsage() const;
	};

	const LookupTables& lookupTables();
}