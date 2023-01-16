#include "stdafx.h"
#include "ConstantBuffer.h"
#include "../D3D/MappedResource.h"
using namespace DirectCompute;

HRESULT ConstantBuffer::create()
{
	if( nullptr == buffer )
	{
		CD3D11_BUFFER_DESC desc{ 16 * 3 * 2, D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE };
		return device()->CreateBuffer( &desc, nullptr, &buffer );
	}
	return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );
}

namespace
{
	__forceinline void copy32( __m128i* rdi, const TensorShape& ts )
	{
		_mm_storeu_si128( rdi, ts.sizeVec() );
		_mm_storeu_si128( rdi + 1, ts.stridesVec() );
	}
}

HRESULT ConstantBuffer::update( const TensorShape& t0 )
{
	MappedResource mapped;
	CHECK( mapped.map( buffer, false ) );

	__m128i* const rdi = ( __m128i* )mapped.data();
	copy32( rdi, t0 );
	return S_OK;
}

HRESULT ConstantBuffer::update( const TensorShape& t0, const TensorShape& t1 )
{
	MappedResource mapped;
	CHECK( mapped.map( buffer, false ) );

	__m128i* const rdi = ( __m128i* )mapped.data();
	copy32( rdi, t0 );
	copy32( rdi + 2, t1 );
	return S_OK;
}

HRESULT ConstantBuffer::update( const TensorShape& t0, const TensorShape& t1, const TensorShape& t2 )
{
	MappedResource mapped;
	CHECK( mapped.map( buffer, false ) );

	__m128i* const rdi = ( __m128i* )mapped.data();
	copy32( rdi, t0 );
	copy32( rdi + 2, t1 );
	copy32( rdi + 4, t2 );
	return S_OK;
}

void ConstantBuffer::bind() const
{
	ID3D11Buffer* p = buffer;
	assert( nullptr != p );
	context()->CSSetConstantBuffers( 0, 1, &p );
}