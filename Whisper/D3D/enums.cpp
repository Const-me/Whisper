#include "stdafx.h"
#include "enums.h"

static const alignas( 16 ) std::array<DXGI_FORMAT, 3> s_tensorViewFormats = { DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT };

DXGI_FORMAT DirectCompute::viewFormat( eDataType dt )
{
	return s_tensorViewFormats[ (uint8_t)dt ];
}