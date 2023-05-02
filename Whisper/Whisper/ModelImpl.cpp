#include "stdafx.h"
#include "ModelImpl.h"
#include "ContextImpl.h"
#include <intrin.h>
#include "../Utils/ReadStream.h"
#include "../modelFactory.h"
using namespace Whisper;

void ModelImpl::FinalRelease()
{
	device.destroy();
}

HRESULT COMLIGHTCALL ModelImpl::createContext( iContext** pp )
{
	auto ts = device.setForCurrentThread();
	ComLight::CComPtr<ComLight::Object<ContextImpl>> obj;

	iModel* m = this;
	CHECK( ComLight::Object<ContextImpl>::create( obj, device, model, m ) );

	obj.detach( pp );
	return S_OK;
}


HRESULT COMLIGHTCALL ModelImpl::tokenize( const char* text, pfnDecodedTokens pfn, void* pv )
{
	std::vector<int> tokens;
	CHECK( model.shared->vocab.tokenize( text, tokens ) );

	if( !tokens.empty() )
		pfn( tokens.data(), (int)tokens.size(), pv );
	else
		pfn( nullptr, 0, pv );

	return S_OK;
}

HRESULT COMLIGHTCALL ModelImpl::clone( iModel** rdi )
{
	if( !device.gpuInfo.cloneableModel() )
	{
		logError( u8"iModel.clone requires the Cloneable model flag" );
		return HRESULT_FROM_WIN32( ERROR_NOT_SUPPORTED );
	}

	ComLight::CComPtr<ComLight::Object<ModelImpl>> obj;
	CHECK( ComLight::Object<ModelImpl>::create( obj, *this ) );
	CHECK( obj->createClone( *this ) );
	obj.detach( rdi );
	return S_OK;
}

HRESULT ModelImpl::createClone( const ModelImpl& source )
{
	auto ts = device.setForCurrentThread();
	CHECK( device.createClone( source.device ) );
	return model.createClone( source.model );
}

HRESULT ModelImpl::load( iReadStream* stm, bool hybrid, const sLoadModelCallbacks* callbacks )
{
	auto ts = device.setForCurrentThread();
	CHECK( device.create( gpuFlags, adapter ) );
	return model.load( stm, hybrid, callbacks );
}

inline bool hasSse41AndF16C()
{
	int cpu_info[ 4 ];
	__cpuid( cpu_info, 1 );

	// https://en.wikipedia.org/wiki/CPUID EAX=1: Processor Info and Feature Bits
	constexpr uint32_t sse41 = ( 1u << 19 );
	constexpr uint32_t f16c = ( 1u << 29 );

#ifdef __AVX__
	constexpr uint32_t requiredBits = sse41 | f16c;
#else
	constexpr uint32_t requiredBits = sse41;
#endif

	const uint32_t ecx = (uint32_t)cpu_info[ 2 ];
	return ( ecx & requiredBits ) == requiredBits;
}

// True when the current CPU is good enough to run the hybrid model
inline bool hasAvxAndFma()
{
	// AVX needs OS support to preserve the 32-bytes registers across context switches, CPU support alone ain't enough
	// Calling a kernel API to check that support
	// The magic number is from there: https://stackoverflow.com/a/35096938/126995
	if( 0 == ( GetEnabledXStateFeatures() & 4 ) )
		return false;

	// FMA3 and F16C
	int cpuInfo[ 4 ];
	__cpuid( cpuInfo, 1 );
	// The magic numbers are from "Feature Information" table on Wikipedia:
	// https://en.wikipedia.org/wiki/CPUID#EAX=1:_Processor_Info_and_Feature_Bits 
	constexpr int requiredBits = ( 1 << 12 ) | ( 1 << 29 );
	if( requiredBits != ( cpuInfo[ 2 ] & requiredBits ) )
		return false;

	// BMI1
	// https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
	__cpuid( cpuInfo, 7 );
	if( 0 == ( cpuInfo[ 1 ] & ( 1 << 3 ) ) )
		return false;

	return true;
}

HRESULT __stdcall Whisper::loadGpuModel( const wchar_t* path, const sModelSetup& setup, const sLoadModelCallbacks* callbacks, iModel** pp )
{
	if( nullptr == path || nullptr == pp )
		return E_POINTER;

	const bool hybrid = setup.impl == eModelImplementation::Hybrid;
	if( hybrid )
	{
#if BUILD_HYBRID_VERSION
		if( !hasAvxAndFma() )
		{
			logError( u8"eModelImplementation.Hybrid model requires a CPU with AVX1, FMA3, F16C and BMI1 support" );
			return ERROR_HV_CPUID_FEATURE_VALIDATION;
		}
#else
		logError( u8"This build of the DLL doesn’t implement eModelImplementation.Hybrid model" );
		return E_NOTIMPL;
#endif
	}
	else if( !hasSse41AndF16C() )
	{
		logError( u8"eModelImplementation.GPU model requires a CPU with SSE 4.1 and F16C support" );
		return ERROR_HV_CPUID_FEATURE_VALIDATION;
	}

	ComLight::Object<ReadStream> stream;
	HRESULT hr = stream.open( path );
	if( FAILED( hr ) )
	{
		logError16( L"Unable to open model binary file \"%s\"", path );
		return hr;
	}

	ComLight::CComPtr<ComLight::Object<ModelImpl>> obj;
	CHECK( ComLight::Object<ModelImpl>::create( obj, setup ) );
	hr = obj->load( &stream, hybrid, callbacks );
	if( FAILED( hr ) )
	{
		logError16( L"Error loading the model from \"%s\"", path );
		return hr;
	}

	obj.detach( pp );
	logInfo16( L"Loaded model from \"%s\" to VRAM", path );
	return S_OK;
}