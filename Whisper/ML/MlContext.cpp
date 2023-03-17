#include "stdafx.h"
#include "MlContext.h"
#include "../D3D/shaderNames.h"
#include "LookupTables.h"
#include "../D3D/shaders.h"
#include "../D3D/Binder.h"
#include "../D3D/MappedResource.h"
#include "../D3D/downloadBuffer.h"
#include "testUtils.h"
#include "reshapedMultiply.h"
using namespace DirectCompute;

// TODO: change this to a field, and set to false when the GPU doesn't support FP64 math
// Most notably, Intel has dropped the support recently:
// https://www.intel.com/content/www/us/en/developer/articles/guide/lp-api-developer-optimization-guide.html#inpage-nav-3-8-undefined
// "To improve power and performance", LOL
constexpr bool usePreciseComputeShaders = true;

MlContext::MlContext( Whisper::ProfileCollection& profileColl ) :
	profiler( profileColl )
{
	check( cb.create() );
	check( profiler.create() );
}

void MlContext::bindShader( eComputeShader cs )
{
	DirectCompute::bindShader( cs );
	profiler.computeShader( cs );
}

void MlContext::mulMatDot( const Tensor& src0, const Tensor& src1, Tensor& res )
{
	const auto& size1 = src1.ne;
	if( 1 != size1[ 3 ] )
		throw E_UNEXPECTED;

	const size_t tempLength = size1[ 0 ] * size1[ 1 ] * size1[ 2 ] * size1[ 3 ];
	const TensorGpuViews& tempBuffer = temp.fp16( tempLength );
	cb.bind();

	bindShader( eComputeShader::mulMatDotReshape );
	cb.update( src1 );
	Binder bind;
	bind.bind( src1, tempBuffer );
	context()->Dispatch( size1[ 1 ], size1[ 2 ], 1 );

	bindShader( eComputeShader::mulMatDotMain );
	cb.update( src0, src1, res );
	bind.bind( src0, tempBuffer, res );

	const auto& size0 = src0.ne;
	// total rows in src0
	const uint32_t nr = size0[ 1 ] * size0[ 2 ] * size0[ 3 ];
	context()->Dispatch( size1[ 1 ], nr, 1 );
}

void MlContext::mulMatMad( const Tensor& a, const Tensor& b, Tensor& res )
{
	// CaptureRaii renderDoc;
	const uint32_t resultElts = res.countElements();
	constexpr uint32_t nth = 4;

	uint32_t fp16;
	TensorGpuViews tempBuffer;

	const eDataType dataType = a.getType();
	if( dataType == eDataType::FP16 )
	{
		fp16 = TRUE;
		tempBuffer = temp.fp16( resultElts * nth );
	}
	else if( dataType == eDataType::FP32 )
	{
		fp16 = FALSE;
		tempBuffer = temp.fp32( resultElts * nth );
	}
	else
		throw E_INVALIDARG;

	TensorShape resultShape = res;
	resultShape.nb = { fp16, resultElts, 0, 0 };

	cb.update( a, b, resultShape );
	bindShader( eComputeShader::mulMatMadMain );
	cb.bind();

	Binder bind;
	bind.bind( { a, b }, { res, tempBuffer } );
	context()->Dispatch( b.ne[ 1 ], b.ne[ 2 ], b.ne[ 3 ] );
}

void MlContext::mulMatTiled( const Tensor& a, const Tensor& b, Tensor& res )
{
	cb.update( a, b, res );
	cb.bind();

	Binder bind;
	bind.bind( a, b, res );

	if( b.ne[ 1 ] == 1 )
	{
		if( b.ne[ 0 ] != 1 )
		{
#if 0
			static PrintUniqueTensorSizes printSize( "mulMatByRow" );
			printSize.print( a, b );
#endif
			// Tensor B is a single row, we have optimized compute shaders for that use case
			// Even 2 of them, tiled and sequential. Select between these two shaders.
			constexpr uint32_t minHeightToTile = 2;
			if( a.ne[ 1 ] < minHeightToTile )
			{
				bindShader( eComputeShader::mulMatByRow );
				context()->Dispatch( a.ne[ 1 ], a.ne[ 2 ], a.ne[ 3 ] );
			}
			else
			{
				bindShader( eComputeShader::mulMatByRowTiled );
				constexpr uint32_t TILE_Y = 64;
				const uint32_t groupsX = ( a.ne[ 1 ] + TILE_Y - 1 ) / TILE_Y;
				context()->Dispatch( groupsX, a.ne[ 2 ], a.ne[ 3 ] );
			}
		}
		else
		{
			// Tensor B is a single element: we have an optimized shader for that as well
			bindShader( eComputeShader::mulMatByScalar );
			context()->Dispatch( a.ne[ 2 ], a.ne[ 3 ], 1 );
		}
	}
	else
	{
		// According to visual studio debugger, when the second argument of this method is a 2D matrix, the first argument is 2D as well.
		// Assuming both arguments are 2D matrices.
		// For optimal VRAM bandwidth utilization, we compute such matrix products in square tiles, a tile is 32x32 elements.
		// Dispatching one thread group for each tile of the output matrix.
		bindShader( eComputeShader::mulMatTiled );

		// These compute shaders correctly handle partial tiles on the right and bottom edges of the output matrix, that's why rounding up
		constexpr uint32_t TILE_SIZE = 32;
		const uint32_t x = ( res.ne[ 0 ] + TILE_SIZE - 1 ) / TILE_SIZE;
		const uint32_t y = ( res.ne[ 1 ] + TILE_SIZE - 1 ) / TILE_SIZE;

		const uint32_t z = res.ne[ 2 ] * res.ne[ 3 ];
		context()->Dispatch( x, y, z );
	}
}

void MlContext::mulMat( const Tensor& src0, const Tensor& src1, Tensor& res )
{
	const uint32_t nb00 = src0.nb[ 0 ];
	const uint32_t nb01 = src0.nb[ 1 ];
	if( nb01 >= nb00 )
		mulMatDot( src0, src1, res );
	else
		mulMatMad( src0, src1, res );
}

namespace
{
	// Must match the HLSL in flashAttention.hlsl
	struct sFlashAttentionConstants
	{
		TensorShape q, k, v, res;
		BOOL masked;
		float scale;
		uint32_t tempBufferStride;
		uint32_t zzPadding;
	};

	struct sFlashAttnDispatchInfo
	{
		uint32_t tempStride;
		uint32_t groupsCount;
	};

	sFlashAttnDispatchInfo makeFlashAttentionConstants( CComPtr<ID3D11Buffer>& buffer, const Tensor& q, const Tensor& k, const Tensor& v, Tensor& res, bool masked )
	{
		if( nullptr == buffer )
		{
			CD3D11_BUFFER_DESC desc{ sizeof( sFlashAttentionConstants ), D3D11_BIND_CONSTANT_BUFFER, D3D11_USAGE_DYNAMIC, D3D11_CPU_ACCESS_WRITE };
			check( device()->CreateBuffer( &desc, nullptr, &buffer ) );
		}

		sFlashAttnDispatchInfo result;

		sFlashAttentionConstants cb;
		cb.q = q;
		cb.k = k;
		cb.v = v;
		cb.res = res;
		cb.masked = masked ? TRUE : FALSE;

		const int neq0 = (int)cb.q.ne[ 0 ];
		const int D = neq0;
		cb.scale = (float)( 1.0 / sqrt( (double)(int)D ) );

		const uint32_t nek1 = cb.k.ne[ 1 ];
		constexpr uint32_t align = 32 / 4;
		result.tempStride = ( ( nek1 + align - 1 ) / align ) * align;
		cb.tempBufferStride = result.tempStride;
		cb.zzPadding = 0;
		result.groupsCount = cb.q.ne[ 1 ] * cb.q.ne[ 2 ] * cb.q.ne[ 3 ];

		MappedResource mapped;
		check( mapped.map( buffer, false ) );
		memcpy( mapped.data(), &cb, sizeof( cb ) );
		return result;
	}
}

void MlContext::flashAttention( const Tensor& q, const Tensor& k, const Tensor& v, Tensor& res, bool masked )
{
	sFlashAttnDispatchInfo di = makeFlashAttentionConstants( flashAttentionConstants, q, k, v, res, masked );

	const uint32_t tempLength = di.tempStride * di.groupsCount;
	const TensorGpuViews& tb = temp.fp32( tempLength );

	csSetCB( flashAttentionConstants );
	ID3D11DeviceContext* const ctx = context();

	Binder bind;
	bind.bind( { q, k, v, lookupTables().exponent()}, {res, tb});

	if constexpr( usePreciseComputeShaders && !enableInexactOptimizations )
	{
		bindShader( eComputeShader::flashAttentionCompat1 );
		ctx->Dispatch( di.groupsCount, 1, 1 );

		bindShader( eComputeShader::flashAttentionCompat2 );
		ctx->Dispatch( ( di.groupsCount + 31 ) / 32, 1, 1 );

		bindShader( eComputeShader::flashAttentionCompat3 );
		ctx->Dispatch( di.groupsCount, 1, 1 );
	}
	else
	{
		// This version is not too bad, e.g. maxAbsDiff = 2.7895e-05, avgDiffSquared = 1.61783e-14
		// And probably much faster.
		// But still, it does not deliver bitwise equality with the reference CPU version
		bindShader( eComputeShader::flashAttention );
		ctx->Dispatch( di.groupsCount, 1, 1 );
	}
}

namespace
{
	// Round up the number to be a multiple of 32
	inline uint32_t roundUp32( uint32_t x )
	{
		return ( x + 31 ) & ( ~31u );
	}
}

void MlContext::convolutionImpl( const Tensor& a, const Tensor& b, Tensor& res, bool is2 )
{
	const uint32_t ne00 = a.ne[ 0 ];
	const uint32_t ne01 = a.ne[ 1 ];
	const uint32_t ne02 = a.ne[ 2 ];

	const uint32_t ne10 = b.ne[ 0 ];
	const uint32_t ne11 = b.ne[ 1 ];

	const uint32_t nb00 = a.nb[ 0 ];
	const uint32_t nb01 = a.nb[ 1 ];
	const uint32_t nb02 = a.nb[ 2 ];

	const uint32_t nb10 = b.nb[ 0 ];
	const uint32_t nb11 = b.nb[ 1 ];

	const uint32_t nb1 = res.nb[ 1 ];

	const uint32_t ew0 = roundUp32( ne01 );

	const uint32_t nk = ne00;
	const uint32_t nh = nk / 2;

	const uint32_t lenTemp1 = ne02 * ew0 * ne00;
	const uint32_t lenTemp2 = ( ne10 + ne00 ) * ew0;

	const TensorGpuViews& temp1 = temp.fp16( lenTemp1, true );
	const TensorGpuViews& temp2 = temp.fp16_2( lenTemp2, true );

	cb.bind();

	bindShader( eComputeShader::convolutionPrep1 );
	cb.update( a );
	Binder bind;
	bind.bind( a, temp1 );
	context()->Dispatch( ne01, ne02, 1 );

	bindShader( eComputeShader::convolutionPrep2 );
	cb.update( a, b );
	bind.bind( b, temp2 );
	context()->Dispatch( ne11, 1, 1 );

	cb.update( a, b, res );
	bind.bind( temp1, temp2, res );
	if( is2 )
	{
		if constexpr( enableInexactOptimizations )
		{
			constexpr uint32_t KERNEL = 3;
			constexpr uint32_t TILE_Y = 8;
			if( a.ne[ 0 ] == KERNEL )
			{
				const uint32_t x = ( ( ne10 / 2 ) + TILE_Y - 1 ) / TILE_Y;
				bindShader( eComputeShader::convolutionMain2Fixed );
				context()->Dispatch( x, ne02, 1 );
				return;
			}
		}
		bindShader( eComputeShader::convolutionMain2 );
		context()->Dispatch( ne10 / 2, ne02, 1 );
	}
	else
	{
		bindShader( eComputeShader::convolutionMain );
		context()->Dispatch( ne10, ne02, 1 );
	}
#if 0
	std::vector<uint16_t> tmp;
	downloadBuffer( temp1, tmp );
	dbgWriteBinaryFile( L"conv-gpu-arg1.bin", tmp.data(), lenTemp1 * 2 );
	downloadBuffer( temp2, tmp );
	dbgWriteBinaryFile( L"conv-gpu-arg2.bin", tmp.data(), lenTemp1 * 2 );
	res.download( tempVector );
	dbgWriteBinaryFile( L"conv-gpu-result.bin", tempVector.data(), tempVector.size() * 4 );
#endif
}

void MlContext::norm( const Tensor& a, Tensor& res )
{
	const uint32_t ne01 = a.ne[ 1 ];
	const uint32_t ne02 = a.ne[ 2 ];
	const uint32_t ne03 = a.ne[ 3 ];

	cb.bind();
	cb.update( a, res );
	Binder bind;
	bind.bind( a, res );

	if constexpr( usePreciseComputeShaders && !enableInexactOptimizations )
	{
		bindShader( eComputeShader::normCompat );
		context()->Dispatch( ( ne01 + 31 ) / 32, ne02, ne03 );
	}
	else
	{
		constexpr uint32_t FIXED_ROW_SIZE = 1024;
		eComputeShader cs = ( a.ne[ 0 ] == FIXED_ROW_SIZE ) ? eComputeShader::normFixed : eComputeShader::norm;
		bindShader( cs );
		context()->Dispatch( ne01, ne02, ne03 );
	}
}

void MlContext::cwiseBinary( const Tensor& a, const Tensor& b, Tensor& res, eComputeShader cs )
{
	assert( isSameShape( a, b ) );
	assert( isSameShape( a, res ) );

	bindShader( cs );
	cb.bind();
	check( cb.update( a, b, res ) );
	Binder bind;
	bind.bind( a, b, res );

	uint32_t rows = a.countRows();
	context()->Dispatch( rows, 1, 1 );
}

Tensor MlContext::add( const Tensor& a, const Tensor& b )
{
	return cwiseBinary( a, b, eComputeShader::add );
}

void MlContext::addInPlace( Tensor& a, const Tensor& b )
{
	if( !isSameShape( a, b ) )
		throw E_INVALIDARG;
	assert( a.getType() == eDataType::FP32 );

	check( cb.update( a, b ) );
	bindShader( eComputeShader::addInPlace );
	cb.bind();

	Binder bind;
	bind.bind( b, a );
	context()->Dispatch( a.ne[ 1 ], a.ne[ 2 ], a.ne[ 3 ] );
}

void MlContext::copyImpl( const Tensor& a, Tensor& res, bool downcastFp32 )
{
	assert( res.isContinuous() );
	const eComputeShader cs = a.isContinuous() ? eComputeShader::copyConvert : eComputeShader::copyTranspose;
	bindShader( cs );

	cb.bind();
	// These two shaders don't need shape of the destination because dense, but they wants a boolean flag whether to implement rounding while downcasting
	TensorShape dummyShape;
	dummyShape.setZero();
	dummyShape.ne[ 0 ] = downcastFp32 ? TRUE : FALSE;
	check( cb.update( a, dummyShape ) );

	Binder bind;
	bind.bind( a, res );
	context()->Dispatch( a.ne[ 1 ], a.ne[ 2 ], a.ne[ 3 ] );
}

namespace
{
	uint32_t bitcast( float val )
	{
		__m128 f = _mm_set_ss( val );
		__m128i i = _mm_castps_si128( f );
		return (uint32_t)_mm_cvtsi128_si32( i );
	}
}

void MlContext::scale( Tensor& a, float mul )
{
	if( !a.isContinuous() )
		throw E_INVALIDARG;

	bindShader( eComputeShader::scaleInPlace );
	cb.bind();
	TensorShape dummyShape;
	dummyShape.setZero();
	dummyShape.ne[ 0 ] = bitcast( mul );
	check( cb.update( a, dummyShape ) );

	Binder bind;
	bind.bind( a );
	context()->Dispatch( a.countRows(), 1, 1 );
}

void MlContext::addRepeat( Tensor& a, const Tensor& b )
{
	check( cb.update( a, b ) );
	bindShader( eComputeShader::addRepeat );
	cb.bind();

	Binder bind;
	bind.bind( b, a );
	context()->Dispatch( a.ne[ 1 ], a.ne[ 2 ], a.ne[ 3 ] );
}

void MlContext::addRepeatScale( Tensor& a, const Tensor& b, float scale )
{
#if 0
	addRepeat( a, b );
	this->scale( a, scale );
	return;
#endif

	TensorShape dummyShape;
	dummyShape.setZero();
	dummyShape.ne[ 0 ] = bitcast( scale );
	check( cb.update( a, b, dummyShape ) );
	bindShader( eComputeShader::addRepeatScale );
	cb.bind();

	Binder bind;
	bind.bind( b, a );
	context()->Dispatch( a.ne[ 1 ], a.ne[ 2 ], a.ne[ 3 ] );
}

void MlContext::fmaRepeat( Tensor& a, const Tensor& mul, const Tensor& add )
{
	eComputeShader cs;
	if( isSameShapeAndLayout( mul, add ) )
	{
		cs = eComputeShader::fmaRepeat1;
		check( cb.update( a, mul ) );
	}
	else
	{
		cs = eComputeShader::fmaRepeat2;
		check( cb.update( a, mul, add ) );
	}

	bindShader( cs );
	cb.bind();
	Binder bind;
	bind.bind( mul, add, a );
	context()->Dispatch( a.ne[ 1 ], a.ne[ 2 ], a.ne[ 3 ] );
}

void MlContext::diagMaskInf( Tensor& a, uint32_t n_past )
{
	if( !a.isContinuous() )
		throw E_INVALIDARG;

	bindShader( eComputeShader::diagMaskInf );
	TensorShape dummyShape;
	dummyShape.setZero();
	dummyShape.ne[ 0 ] = n_past;

	cb.bind();
	check( cb.update( a, dummyShape ) );

	Binder bind;
	bind.bind( a );

	const uint32_t n = a.countRows();
	const uint32_t nr = a.ne[ 1 ];
	const uint32_t nz = n / nr;
	context()->Dispatch( nr, nz, 1 );
}

void MlContext::softMax( Tensor& a, float inputScale )
{
	if( !a.isContinuous() )
		throw E_INVALIDARG;

	if constexpr( usePreciseComputeShaders && !enableInexactOptimizations )
	{
		assert( inputScale == 1.0f );
		bindShader( eComputeShader::softMaxCompat );
		const uint32_t nr = a.countRows();
		TensorShape dummyShape;
		dummyShape.setZero();
		dummyShape.ne[ 0 ] = nr;

		cb.bind();
		check( cb.update( a, dummyShape ) );

		Binder bind;
		bind.bind( lookupTables().exponent(), a);
		context()->Dispatch( ( nr + 31 ) / 32, 1, 1 );
	}
	else
	{
#if 0
		static PrintUniqueTensorSizes printSizes( "softMax" );
		printSizes.print( a );
#endif
		constexpr uint32_t FIXED_ROW_SIZE = 1500;

		eComputeShader cs;
		if( a.ne[ 0 ] == FIXED_ROW_SIZE )
			cs = eComputeShader::softMaxFixed;
		else if( a.ne[ 0 ] >= ( 1024 * 4 ) )
			cs = eComputeShader::softMaxLong;
		else
			cs = eComputeShader::softMax;

		bindShader( cs );
		const uint32_t nr = a.countRows();
		TensorShape dummyShape;
		dummyShape.setZero();
		dummyShape.ne[ 0 ] = nr;
		dummyShape.ne[ 1 ] = bitcast( inputScale );

		cb.bind();
		check( cb.update( a, dummyShape ) );

		Binder bind;
		bind.bind( lookupTables().exponent(), a);
		context()->Dispatch( nr, 1, 1 );
	}
}

void MlContext::addRepeatGelu( Tensor& a, const Tensor& b )
{
	check( cb.update( a, b ) );
	bindShader( eComputeShader::addRepeatGelu );
	cb.bind();

	Binder bind;
	bind.bind( b, lookupTables().gelu(), a);
	context()->Dispatch( a.ne[ 1 ], a.ne[ 2 ], a.ne[ 3 ] );
}

namespace
{
	inline bool canAddRows( const Tensor& tokenEmbedding, const Tensor& positionalEmbedding, const Tensor& embd, uint32_t pastTokensCount )
	{
		if( tokenEmbedding.ne[ 0 ] != positionalEmbedding.ne[ 0 ] )
			return false;	// Different row lengths
		if( embd.ne[ 0 ] + pastTokensCount > positionalEmbedding.ne[ 1 ] )
			return false;	// Too many rows requested, positionalEmbedding matrix doesn't have that many 
		return true;
	}
}

Tensor MlContext::addRows( const Tensor& tokenEmbedding, const Tensor& positionalEmbedding, const Tensor& embd, uint32_t pastTokensCount )
{
	if( !canAddRows( tokenEmbedding, positionalEmbedding, embd, pastTokensCount ) )
		throw E_INVALIDARG;

	const uint32_t rowLength = tokenEmbedding.ne[ 0 ];
	const uint32_t rows = embd.ne[ 0 ];
	Tensor result = createTensor( eDataType::FP32, { rowLength, rows } );

	TensorShape constants;
	// rowLength
	constants.ne[ 0 ] = rowLength;
	// pastTokensCount
	constants.ne[ 1 ] = pastTokensCount;
	// outputRowStride
	constants.ne[ 2 ] = result.nb[ 1 ];
	// embStrides
	constants.nb[ 0 ] = tokenEmbedding.nb[ 0 ];
	constants.nb[ 1 ] = tokenEmbedding.nb[ 1 ];
	// posStrides
	constants.nb[ 2 ] = positionalEmbedding.nb[ 0 ];
	constants.nb[ 3 ] = positionalEmbedding.nb[ 1 ];
	check( cb.update( constants ) );

	bindShader( eComputeShader::addRows );
	cb.bind();
	Binder bind;
	bind.bind( { tokenEmbedding, positionalEmbedding, embd }, { result } );
	context()->Dispatch( rows, 1, 1 );
	return result;
}

Tensor MlContext::reshapePanels( const Tensor& a )
{
	constexpr uint32_t TILE_SIZE = ReshapedMultiply::TILE_SIZE;

	const eDataType dataType = a.getType();
	// Reshaping into column major horizontal panels, height = TILE_SIZE, width = width of the source matrix

	// Round height to multiple of tile size
	std::array<uint32_t, 4> ne = a.ne;
	// Dispatch a group of threads thread per panel
	const uint32_t groupsX = ( ne[ 1 ] + TILE_SIZE - 1 ) / TILE_SIZE;
	ne[ 1 ] = groupsX * TILE_SIZE;;
	// Each panel has [ size.x, TILE_SIZE ] elements
	const uint32_t panelSize = ne[ 0 ] * TILE_SIZE;

	Tensor result = createTensor( dataType, ne );

	TensorShape constants;
	constants.setZero();
	// uint panelSize : packoffset( c2.y );
	constants.ne[ 1 ] = panelSize;
	// uint2 layerStrides: packoffset( c2.z );
	constants.ne[ 2 ] = result.nb[ 2 ];
	constants.ne[ 3 ] = result.nb[ 3 ];

	check( cb.update( a, constants ) );
	bindShader( eComputeShader::matReshapePanels );
	cb.bind();

	Binder bind;
	bind.bind( a, result );
	context()->Dispatch( groupsX, a.ne[ 2 ], a.ne[ 3 ] );

#if 0
	if( dataType == eDataType::FP32 )
	{
		std::vector<float> v1, v2;
		a.download( v1 );
		result.download( v2 );
		__debugbreak();
	}
	else if( dataType == eDataType::FP16 )
	{
		std::vector<uint16_t> v1, v2;
		a.download( v1 );
		result.download( v2 );
		__debugbreak();
	}
#endif

	// Set up size and stride expected by the mulMatTiledEx compute shader
	result.ne = a.ne;
	result.nb[ 0 ] = 0;
	result.nb[ 1 ] = panelSize;
	return result;
}

Tensor MlContext::mulMatTiledEx( const Tensor& a, const Tensor& b )
{
	constexpr uint32_t TILE_SIZE = ReshapedMultiply::TILE_SIZE;

	if( !canMulMat( a, b ) )
		throw E_INVALIDARG;	// Wrong size
	if( 0 != ( a.nb[ 0 ] | b.nb[ 0 ] ) )
		throw E_INVALIDARG;	// Both tensors are expected to be pre-transposed into these panels

	Tensor res = createTensor( eDataType::FP32, { a.ne[ 1 ], b.ne[ 1 ], a.ne[ 2 ], b.ne[ 3 ] } );

	check( cb.update( a, b, res ) );
	bindShader( eComputeShader::mulMatTiledEx );
	cb.bind();

	Binder bind;
	bind.bind( a, b, res );

	const uint32_t x = ( res.ne[ 0 ] + TILE_SIZE - 1 ) / TILE_SIZE;
	const uint32_t y = ( res.ne[ 1 ] + TILE_SIZE - 1 ) / TILE_SIZE;
	const uint32_t z = res.ne[ 2 ] * res.ne[ 3 ];
	context()->Dispatch( x, y, z );

	return res;
}

Tensor MlContext::mulMatByRowTiledEx( const Tensor& a, const Tensor& b )
{
	constexpr uint32_t TILE_SIZE = ReshapedMultiply::TILE_SIZE;
	assert( canMulMat( a, b ) );
	assert( b.ne[ 1 ] == 1 );

	Tensor res = createTensor( eDataType::FP32, { a.ne[ 1 ], 1, a.ne[ 2 ], b.ne[ 3 ] } );

	check( cb.update( a, b, res ) );
	bindShader( eComputeShader::mulMatByRowTiledEx );
	cb.bind();

	Binder bind;
	bind.bind( a, b, res );

	const uint32_t x = ( res.ne[ 0 ] + TILE_SIZE - 1 ) / TILE_SIZE;
	const uint32_t y = res.ne[ 2 ];
	const uint32_t z = res.ne[ 3 ];
	context()->Dispatch( x, y, z );

	return res;
}

void MlContext::addRepeatEx( Tensor& dest, const Tensor& pattern, const Tensor& finalAdd )
{
	if( !isSameShape( dest, finalAdd ) )
		throw E_INVALIDARG;
	assert( dest.getType() == eDataType::FP32 );

	check( cb.update( dest, pattern, finalAdd ) );
	bindShader( eComputeShader::addRepeatEx );
	cb.bind();

	Binder bind;
	bind.bind( pattern, finalAdd, dest );
	context()->Dispatch( dest.ne[ 1 ], dest.ne[ 2 ], dest.ne[ 3 ] );
}

__m128i MlContext::getMemoryUse() const
{
	__m128i v = cb.getMemoryUse();
	v = _mm_add_epi64( v, temp.getMemoryUse() );
	v = _mm_add_epi64( v, bufferMemoryUsage( flashAttentionConstants ) );
	v = _mm_add_epi64( v, lookupTables().getMemoryUsage());
	return v;
}