#pragma once

#define CHECK( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) return __hr; }
#define CHECK_LOG( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) { logErrorHr(__hr, u8"%s failed", #hr ); return __hr; } }

inline void check( HRESULT hr )
{
	if( SUCCEEDED( hr ) )
		return;
	throw hr;
}

inline __m128i __vectorcall load16( const int* rsi )
{
	return _mm_loadu_si128( ( const __m128i* )rsi );
}
inline __m128i __vectorcall load16( const uint32_t* rsi )
{
	return _mm_loadu_si128( ( const __m128i* )rsi );
}
inline __m128i __vectorcall load( const std::array<uint32_t, 4>& arr )
{
	return load16( arr.data() );
}
inline void __vectorcall store16( void* rdi, __m128i v )
{
	_mm_storeu_si128( ( __m128i* )rdi, v );
}
inline void __vectorcall store12( void* rdi, __m128i v )
{
	_mm_storel_epi64( ( __m128i* )rdi, v );
	( (int*)rdi )[ 2 ] = _mm_extract_epi32( v, 2 );
}
inline void __vectorcall store( std::array<uint32_t, 4>& arr, __m128i v )
{
	store16( arr.data(), v );
}
inline bool __vectorcall vectorEqual( __m128i a, __m128i b )
{
	__m128i xx = _mm_xor_si128( a, b );
	return (bool)_mm_testz_si128( xx, xx );
}

inline __m128i __vectorcall setLow_size( size_t low )
{
	return _mm_cvtsi64_si128( (int64_t)low );
}
inline __m128i __vectorcall setr_size( size_t low, size_t high )
{
	__m128i v = setLow_size( low );
	v = _mm_insert_epi64( v, (int64_t)high, 1 );
	return v;
}
inline __m128i __vectorcall setHigh_size( size_t high )
{
	__m128i v = _mm_setzero_si128();
	v = _mm_insert_epi64( v, (int64_t)high, 1 );
	return v;
}

void setCurrentThreadName( const char* name );

inline HRESULT getLastHr()
{
	return HRESULT_FROM_WIN32( GetLastError() );
}

// Scale time in seconds from unsigned 64 bit rational number ( mul / div ) into 100-nanosecond ticks
// These 100-nanosecond ticks are used in NTFS, FILETIME, .NET standard library, media foundation, and quite a few other places
inline uint64_t makeTime( uint64_t mul, uint64_t div )
{
	mul *= 10'000'000;
	mul += ( ( div / 2 ) - 1 );
	return mul / div;
}

template<class E>
inline size_t vectorMemoryUse( const std::vector<E>& vec )
{
	return sizeof( E ) * vec.capacity();
}

// The formula is pow( mul / div, -0.25 )
float computeScaling( int mul, int div );