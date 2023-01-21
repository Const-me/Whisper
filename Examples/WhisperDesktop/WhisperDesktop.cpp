#include "stdafx.h"
#include "AppState.h"
#include "Utils/miscUtils.h"
#include "LoadModelDlg.h"
#include "TranscribeDlg.h"
#include "CaptureDlg.h"

static HRESULT dialogLoadModel( AppState& appState )
{
	LoadModelDlg loadDialog{ appState };
	HRESULT hr = loadDialog.show();
	if( FAILED( hr ) )
	{
		reportFatalError( "Error loading the model", hr );
		return hr;
	}
	appState.automaticallyLoadModel = false;
	return hr;
}

static HRESULT dialogTranscribe( AppState& appState )
{
	TranscribeDlg dialog{ appState };
	return dialog.show();
}

static HRESULT dialogCapture( AppState& appState )
{
	CaptureDlg dialog{ appState };
	return dialog.show();
}

using pfnDialog = HRESULT( * )( AppState& appState );
static const std::array<pfnDialog, 4> s_dialogs =
{
	nullptr, // S_OK
	&dialogLoadModel,   // SCREEN_MODEL
	&dialogTranscribe,  // SCREEN_TRANSCRIBE
	&dialogCapture,	    // SCREEN_CAPTURE
};

int __stdcall wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
	AppState appState;
	HRESULT hr = appState.startup();
	if( FAILED( hr ) )
		return hr;

	appState.findModelSource();

	hr = SCREEN_MODEL;
	while( true )
	{
		pfnDialog pfn = s_dialogs[ hr ];
		if( nullptr == pfn )
			return S_OK;
		hr = pfn( appState );
		if( FAILED( hr ) )
			return hr;
		if( hr == SCREEN_MODEL )
			appState.model = nullptr;
	}
}