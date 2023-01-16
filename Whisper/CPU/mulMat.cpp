#include "stdafx.h"
#include "mulMat.h"
#include "mulMatImpl.h"
using namespace CpuCompute;

namespace
{
	template<uint8_t panelHeightRegs, uint8_t tileWidthFloats>
	static HRESULT mulMatImpl( Tensor& result, const Tensor& a, const Tensor& b, ParallelForRunner& pfor )
	{
		MulMatImpl<panelHeightRegs, tileWidthFloats> impl{ result, a, b, pfor };
		return impl.run( pfor );
	}
}

HRESULT CpuCompute::mulMat( Tensor& result, const Tensor& a, const Tensor& b, ParallelForRunner& pfor )
{
	if( a.type() != eDataType::FP16 )
		return E_NOTIMPL;
	if( b.type() != eDataType::FP32 )
		return E_NOTIMPL;

	// return mulMatImpl<1, 1>( result, a, b, pfor );

	if( b.ne[ 1 ] == 1 )
	{
		// Multiplying by a single row
		if( a.ne[ 1 ] >= 32 )
			return mulMatImpl<4, 1>( result, a, b, pfor );
		else
			return mulMatImpl<1, 1>( result, a, b, pfor );
	}
	else if( b.ne[ 1 ] == 2 )
	{
		if( a.ne[ 1 ] >= 32 )
			return mulMatImpl<4, 2>( result, a, b, pfor );
		else
			return mulMatImpl<1, 2>( result, a, b, pfor );
	}
	else if( b.ne[ 1 ] == 3 )
	{
		if( a.ne[ 1 ] >= 16 )
			return mulMatImpl<2, 3>( result, a, b, pfor );
		else
			return mulMatImpl<1, 3>( result, a, b, pfor );
	}
	else
	{
		if( a.ne[ 1 ] >= 16 )
			return mulMatImpl<2, 4>( result, a, b, pfor );
		else
			return mulMatImpl<1, 4>( result, a, b, pfor );
	}
}