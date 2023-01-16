#pragma once
#include <type_traits>

// Calling conventions
#ifdef _MSC_VER
#define COMLIGHTCALL __stdcall
#define DECLSPEC_NOVTABLE   __declspec(novtable)
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__)
#define COMLIGHTCALL __attribute__((stdcall))
#else
#define COMLIGHTCALL
#endif
#define DECLSPEC_NOVTABLE
#else
#error Unsupported C++ compiler
#endif

#include "utils/guid_parse.hpp"

#define DEFINE_INTERFACE_ID( guidString ) static constexpr GUID iid() { return ::ComLight::make_guid( guidString ); }

namespace ComLight
{
	// This thing is binary compatible with IUnknown from Windows SDK. See DesktopClient demo project, it uses normal COM interop in .NET framework 4.7 to call my implementation.
	struct DECLSPEC_NOVTABLE IUnknown
	{
		DEFINE_INTERFACE_ID( "00000000-0000-0000-c000-000000000046" );

		virtual HRESULT COMLIGHTCALL QueryInterface( REFIID riid, void **ppvObject ) = 0;

		virtual uint32_t COMLIGHTCALL AddRef() = 0;

		virtual uint32_t COMLIGHTCALL Release() = 0;
	};
}