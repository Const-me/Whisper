#include "stdafx.h"
#include "ProfileCollection.h"
#include "GpuProfiler.h"
#include "../Whisper/WhisperModel.h"
#include "../D3D/shaderNames.h"
using namespace Whisper;

ProfileCollection::Measure& ProfileCollection::measure( DirectCompute::eProfilerBlock which )
{
	uint32_t key = (uint16_t)which;
	key |= 0x20000;
	return measures[ key ];
}

ProfileCollection::Measure& ProfileCollection::measure( DirectCompute::eComputeShader which )
{
	uint32_t key = (uint16_t)which;
	key |= 0x30000;
	return measures[ key ];
}

ProfileCollection::Measure& ProfileCollection::measure( eCpuBlock which )
{
	uint32_t key = (uint8_t)which;
	key |= 0x10000;
	CComCritSecLock<CComAutoCriticalSection> lock{ critSec };
	return measures[ key ];
}

#if PROFILER_COLLECT_TAGS
ProfileCollection::Measure& ProfileCollection::measure( DirectCompute::eComputeShader which, uint16_t tag )
{
	uint32_t key = (uint8_t)which;
	key = key << 16;
	key |= tag;
	CComCritSecLock<CComAutoCriticalSection> lock{ critSec };
	return taggedShaders[ key ];
}
#endif

namespace
{
	using pfnPrintEnum = const char* ( * )( uint16_t val );

	static const char* printCpuBlock( uint16_t id )
	{
		const eCpuBlock which = (eCpuBlock)id;
		switch( which )
		{
#define V(x) case eCpuBlock::x: return #x
			V( LoadModel );
			V( RunComplete );
			V( Run );
			V( Callbacks );
			V( Spectrogram );
			V( Sample );
			V( VAD );
			V( Encode );
			V( Decode );
			V( DecodeStep );
			V( DecodeLayer );
#undef V
		}
		assert( false );
		return nullptr;
	}

	static const char* printGpuBlock( uint16_t id )
	{
		using DirectCompute::eProfilerBlock;
		const eProfilerBlock which = (eProfilerBlock)id;

		switch( which )
		{
#define V(x) case eProfilerBlock::x: return #x
			V( LoadModel );
			V( Run );
			V( Encode );
			V( EncodeLayer );
			V( Decode );
			V( DecodeStep );
			V( DecodeLayer );
#undef V
		}
		assert( false );
		return nullptr;
	}

	static const char* printShader( uint16_t id )
	{
		return DirectCompute::computeShaderName( (DirectCompute::eComputeShader)id );
	}

	static pfnPrintEnum printSectionStart( uint16_t type )
	{
		switch( type )
		{
		case 1:
			logInfo( u8"    CPU Tasks" );
			return &printCpuBlock;
		case 2:
			logInfo( u8"    GPU Tasks" );
			return &printGpuBlock;
		case 3:
			logInfo( u8"    Compute Shaders" );
			return &printShader;
		default:
			return nullptr;
		}
	}

	struct PrintedTime
	{
		double value;
		const char* unit;

		PrintedTime( uint64_t ticks )
		{
			const double dbl = (double)(int64_t)ticks;
			if( ticks >= 10'000'000 )
			{
				value = dbl / 1.0E+7;
				unit = "seconds";
			}
			else if( ticks >= 10'000 )
			{
				value = dbl / 1.0E+4;
				unit = "milliseconds";
			}
			else
			{
				value = dbl / 1.0E+1;
				unit = "microseconds";
			}
		}
		PrintedTime( double dbl )
		{
			if( dbl >= 10'000'000 )
			{
				value = dbl / 1.0E+7;
				unit = "seconds";
			}
			else if( dbl >= 10'000 )
			{
				value = dbl / 1.0E+4;
				unit = "milliseconds";
			}
			else
			{
				value = dbl / 1.0E+1;
				unit = "microseconds";
			}
		}
	};
}

void ProfileCollection::Measure::print( const char* name ) const
{
	PrintedTime total{ totalTicks };
	if( 1 == count )
		logInfo( u8"%s\t%g %s", name, total.value, total.unit );
	else
	{
		PrintedTime avg = (double)totalTicks / (double)(int64_t)count;
		logInfo( u8"%s\t%g %s, %zu calls, %g %s average", name, total.value, total.unit, count, avg.value, avg.unit );
	}
}

#if PROFILER_COLLECT_TAGS
struct TaggedShaderCmp
{
	bool operator()( uint16_t cs, uint32_t key ) const
	{
		return cs < key >> 16;
	}
	bool operator()( uint32_t key, uint16_t cs ) const
	{
		return key >> 16 < cs;
	}
};

void ProfileCollection::TaggedTemp::print() const
{
	PrintedTime total{ ticks };
	if( 1 == count )
		logInfo( u8"  %s\t%g %s", name, total.value, total.unit );
	else
	{
		PrintedTime avg = (double)ticks / (double)(int64_t)count;
		logInfo( u8"  %s\t%g %s, %zu calls, %g %s average", name, total.value, total.unit, count, avg.value, avg.unit );
	}
}
#endif

void ProfileCollection::print()
{
	keysTemp.clear();
	for( POSITION pos = measures.GetStartPosition(); nullptr != pos; )
	{
		auto* p = measures.GetNext( pos );
		if( p->m_value.count == 0 )
			continue;
		keysTemp.push_back( p->m_key );
	}

	std::sort( keysTemp.begin(), keysTemp.end() );
	auto it = std::lower_bound( keysTemp.begin(), keysTemp.end(), 0x30000u );
	if( it != keysTemp.end() )
	{
		auto lambda = [ this ]( uint32_t a, uint32_t b )
		{
			const uint64_t ta = measures.Lookup( a )->m_value.totalTicks;
			const uint64_t tb = measures.Lookup( b )->m_value.totalTicks;
			return ta > tb;
		};
		std::stable_sort( it, keysTemp.end(), lambda );
	}

#if PROFILER_COLLECT_TAGS
	taggedKeysTemp.clear();
	for( POSITION pos = taggedShaders.GetStartPosition(); nullptr != pos; )
	{
		auto* p = taggedShaders.GetNext( pos );
		if( p->m_value.count == 0 )
			continue;
		taggedKeysTemp.push_back( p->m_key );
	}
	std::sort( taggedKeysTemp.begin(), taggedKeysTemp.end() );
#endif

	uint16_t prevKeyType = 0;
	pfnPrintEnum pfn = nullptr;
	for( uint32_t k : keysTemp )
	{
		const uint16_t type = (uint16_t)( k >> 16 );
		if( type != prevKeyType )
		{
			prevKeyType = type;
			pfn = printSectionStart( type );
		}
		if( pfn == nullptr )
			continue;
		const auto* p = measures.Lookup( k );
		assert( nullptr != p );
		p->m_value.print( pfn( (uint16_t)k ) );

#if PROFILER_COLLECT_TAGS
		if( type == 3 )	
		{
			// Compute shader
			auto range = std::equal_range( taggedKeysTemp.begin(), taggedKeysTemp.end(), (uint16_t)k, TaggedShaderCmp{} );
			if( range.first != range.second )
			{
				// We have at least 1 tag for that compute shader
				taggedTimes.clear();
				uint64_t totalTicks = 0;
				size_t totalCount = 0;
				for( auto it = range.first; it != range.second; it++ )
				{
					const uint32_t key = *it;
					const uint16_t tagId = (uint16_t)key;
					assert( 0 != tagId );
					const auto* p = taggedShaders.Lookup( key );
					assert( nullptr != p );

					auto& rdi = taggedTimes.emplace_back();
					rdi.ticks = p->m_value.totalTicks;
					totalTicks += p->m_value.totalTicks;

					rdi.count = p->m_value.count;
					totalCount += p->m_value.count;

					rdi.name = tagNames[ tagId ];
				}

				assert( totalCount <= p->m_value.count );
				if( totalCount < p->m_value.count )
				{
					auto& rdi = taggedTimes.emplace_back();
					rdi.ticks = p->m_value.totalTicks - totalTicks;
					rdi.count = p->m_value.count - totalCount;
					rdi.name = tagNames[ 0 ];
				}
				std::stable_sort( taggedTimes.begin(), taggedTimes.end() );
				for( const auto& e : taggedTimes )
					e.print();
			}
		}
#endif
	}
}

void ProfileCollection::reset()
{
	for( POSITION pos = measures.GetStartPosition(); nullptr != pos; )
		measures.GetNextValue( pos ).reset();
}

ProfileCollection::ProfileCollection( const WhisperModel& model )
{
	const __m128i vals = model.getLoadTimes();

	uint64_t s = (uint64_t)_mm_cvtsi128_si64( vals );
	measure( eCpuBlock::LoadModel ).add( s );

	s = (uint64_t)_mm_extract_epi64( vals, 1 );
	measure( DirectCompute::eProfilerBlock::LoadModel ).add( s );
#if PROFILER_COLLECT_TAGS
	// Tag ID 0 means no tag at all. makeTagId() method returns 0 for nullptr name, and starts numbering with 1 for non-empoty tag names
	// Push the tag name corresponding to ID = 0, this way we can index directly with tag IDs.
	tagNames.push_back( "<untagged>" );
#endif
}

uint16_t ProfileCollection::makeTagId( const char* tag )
{
#if PROFILER_COLLECT_TAGS
	if( nullptr == tag )
		return 0;
	auto p = tagIDs.Lookup( tag );
	if( nullptr != p )
		return p->m_value;
	const size_t newTag = tagIDs.GetCount() + 1;
	if( newTag <= 0xFFFF )
	{
		tagIDs.SetAt( tag, (uint16_t)newTag );
		tagNames.push_back( tag );
		return (uint16_t)newTag;
	}
	throw DISP_E_OVERFLOW;
#else
	return 0;
#endif
}