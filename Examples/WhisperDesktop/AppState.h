#pragma once
#include "Utils/DebugConsole.h"

class AppState
{
	bool coInit = false;
	CRegKey registryKey;
	CIcon appIcon;
public:

	struct ModelSource
	{
		CString path;
		bool found = false;
		Whisper::eModelImplementation impl = (Whisper::eModelImplementation)0;
		uint64_t sizeInBytes = 0;
	};
	ModelSource source;

	DebugConsole console;
	CComPtr<Whisper::iMediaFoundation> mediaFoundation;
	CComPtr<Whisper::iModel> model;

	~AppState();

	// Setup the initial things
	HRESULT startup();

	HRESULT findModelSource();

	HRESULT saveModelSource();

	uint32_t languageRead();
	void languageWrite( uint32_t key );

	CString stringLoad( LPCTSTR name );
	void stringStore( LPCTSTR name, LPCTSTR value );
	uint32_t dwordLoad( LPCTSTR name, uint32_t fallback );
	void dwordStore( LPCTSTR name, uint32_t value );
	bool boolLoad( LPCTSTR name );
	void boolStore( LPCTSTR name, bool val );

	bool automaticallyLoadModel = true;

	void lastScreenSave( HRESULT code );
	HRESULT lastScreenLoad();

	void setupIcon( CWindow* wnd );

	uint32_t gpuFlagsLoad();
	void gpuFlagsStore( uint32_t flags );
};

constexpr HRESULT SCREEN_MODEL = 1;
constexpr HRESULT SCREEN_TRANSCRIBE = 2;
constexpr HRESULT SCREEN_CAPTURE = 3;