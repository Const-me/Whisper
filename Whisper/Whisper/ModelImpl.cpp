#include "stdafx.h"
#include "ModelImpl.h"
#include "../ML/mlStartup.h"
#include "ContextImpl.h"
#include <intrin.h>
#include "../Utils/ReadStream.h"
#include "../modelFactory.h"
using namespace Whisper;

namespace
{
	volatile long s_refCounter = 0;
}

HRESULT ModelImpl::FinalConstruct()
{
	if( 1 != InterlockedIncrement( &s_refCounter ) )
		return S_FALSE;
	return DirectCompute::mlStartup( gpuFlags );
}

void ModelImpl::FinalRelease()
{
	if( 0 == InterlockedDecrement( &s_refCounter ) )
		DirectCompute::mlShutdown();
}

HRESULT COMLIGHTCALL ModelImpl::createContext( iContext** pp )
{
	ComLight::CComPtr<ComLight::Object<ContextImpl>> obj;

	iModel* m = this;
	CHECK( ComLight::Object<ContextImpl>::create( obj, model, m ) );

	obj.detach( pp );
	return S_OK;
}

HRESULT ModelImpl::load( iReadStream* stm, bool hybrid, const sLoadModelCallbacks* callbacks )
{
	return model.load( stm, hybrid, callbacks );
}

inline bool hasSse41()
{
	int cpu_info[ 4 ];
	__cpuid( cpu_info, 1 );
	return ( cpu_info[ 2 ] & ( 1 << 19 ) ) != 0;
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

HRESULT __stdcall Whisper::loadGpuModel( const wchar_t* path, bool hybrid, uint32_t flags, const sLoadModelCallbacks* callbacks, iModel** pp )
{
	if( nullptr == path || nullptr == pp )
		return E_POINTER;

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
	else if( !hasSse41() )
	{
		logError( u8"eModelImplementation.GPU model requires a CPU with SSE 4.1 support" );
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
	CHECK( ComLight::Object<ModelImpl>::create( obj, flags ) );
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