#include "stdafx.h"
#include "LoadModelDlg.h"
#include "Utils/miscUtils.h"
#include "Utils/logger.h"
#include "ModelAdvancedDlg.h"

constexpr int progressMaxInteger = 1024 * 8;

HRESULT LoadModelDlg::show()
{
	auto res = DoModal( nullptr );
	if( res == -1 )
		return HRESULT_FROM_WIN32( GetLastError() );
	if( res == IDOK )
	{
		HRESULT hr = appState.lastScreenLoad();
		switch( hr )
		{
		case SCREEN_TRANSCRIBE:
		case SCREEN_CAPTURE:
			return hr;
		default:
			return SCREEN_TRANSCRIBE;
		}
	}
	return S_OK;
}

LRESULT LoadModelDlg::OnInitDialog( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled )
{
	// First DDX call, hooks up variables to controls.
	DoDataExchange( false );

	cbConsole.initialize( m_hWnd, IDC_CONSOLE, appState );
	implPopulateCombobox( cbModelType, appState.source.impl );
	modelPath.SetWindowTextW( appState.source.path );

	HRESULT hr = work.create( this );
	if( FAILED( hr ) )
	{
		CString text = L"CreateThreadpoolWork failed\n";
		text += formatErrorMessage( hr );
		::MessageBox( m_hWnd, text, L"Unable to load the model", MB_OK | MB_ICONWARNING );
		return TRUE;
	}

	editorsWindows.reserve( 6 );
	editorsWindows = { modelPath, cbModelType, GetDlgItem( IDC_BROWSE ), GetDlgItem( IDC_MODEL_ADV ), GetDlgItem( IDOK ), GetDlgItem( IDCANCEL ) };
	pendingWindows.reserve( 2 );
	pendingWindows = { GetDlgItem( IDC_PENDING_TEXT ), progressBar };

	progressBar.SetRange32( 0, progressMaxInteger );
	progressBar.SetStep( 1 );

	appState.setupIcon( this );
	ATLVERIFY( CenterWindow() );
	if( !appState.source.found || !appState.automaticallyLoadModel )
		return 0;

	// AppState.findModelSource() method has located model parameters in registry;
	// Post a notification identical to the "OK" button click event.
	PostMessage( WM_COMMAND, IDOK, (LPARAM)( GetDlgItem( IDOK ).m_hWnd ) );

	return 0;
}

LRESULT LoadModelDlg::OnBrowse( UINT, INT, HWND, BOOL& bHandled )
{
	bHandled = TRUE;

	CString path;
	modelPath.GetWindowText( path );
	if( !getOpenFileName( m_hWnd, L"Select a GGML Model File", L"Binary files (*.bin)\0*.bin\0\0", path ) )
		return 0;

	modelPath.SetWindowText( path );
	appState.source.path = path;
	return 0;
}

LRESULT LoadModelDlg::validationError( LPCTSTR message )
{
	reportError( m_hWnd, message, L"Unable to load the model" );
	return 0;
}

LRESULT LoadModelDlg::validationError( LPCTSTR message, HRESULT hr )
{
	reportError( m_hWnd, message, L"Unable to load the model", hr );
	return 0;
}

void LoadModelDlg::setPending( bool nowPending )
{
	const BOOL enable = nowPending ? FALSE : TRUE;
	for( HWND w : editorsWindows )
		::EnableWindow( w, enable );

	const int show = nowPending ? SW_NORMAL : SW_HIDE;
	for( HWND w : pendingWindows )
		::ShowWindow( w, show );

	if( nowPending )
		progressBar.SetMarquee( TRUE, 0 );
	else
		progressBar.SetMarquee( FALSE, 0 );
}

LRESULT LoadModelDlg::OnOk( UINT, INT, HWND, BOOL& bHandled )
{
	modelPath.GetWindowText( path );
	if( path.GetLength() <= 0 )
		return validationError( L"Please select a model GGML file" );

	{
		CAtlFile file;
		HRESULT hr = file.Create( path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING );
		if( FAILED( hr ) )
			return validationError( L"Unable to open the model file", hr );

		ULONGLONG cb = 0;
		file.GetSize( cb );
		appState.source.sizeInBytes = cb;
	}

	impl = implGetValue( cbModelType );
	if( impl == (Whisper::eModelImplementation)0 )
		return validationError( L"Please select a model type" );

	setPending( true );
	work.post();
	return 0;
}

static HRESULT loadModel( const wchar_t* path, Whisper::sModelSetup setup, const Whisper::sLoadModelCallbacks* callbacks, Whisper::iModel** pp )
{
	constexpr bool dbgTestClone = false;

	if constexpr( dbgTestClone )
		setup.flags |= (uint32_t)Whisper::eGpuModelFlags::Cloneable;

	CComPtr<Whisper::iModel> res;
	CHECK( Whisper::loadModel( path, setup, callbacks, &res ) );

	if constexpr( dbgTestClone )
	{
		CComPtr<Whisper::iModel> clone;
		CHECK( res->clone( &clone ) );
		*pp = clone.Detach();
		return S_OK;
	}
	else
	{
		*pp = res.Detach();
		return S_OK;
	}
}

void __stdcall LoadModelDlg::poolCallback() noexcept
{
	CComPtr<Whisper::iModel> model;
	clearLastError();
	loadError = L"";
	Whisper::sLoadModelCallbacks lmcb;
	lmcb.cancel = nullptr;
	lmcb.progress = &LoadModelDlg::progressCallback;
	lmcb.pv = this;
	Whisper::sModelSetup setup;
	setup.impl = impl;
	setup.flags = appState.gpuFlagsLoad();
	CString adapter = appState.stringLoad( L"gpu" );
	setup.adapter = adapter;
	HRESULT hr = ::loadModel( path, setup, &lmcb, &model );
	if( SUCCEEDED( hr ) )
		appState.model = model;
	else
		getLastError( loadError );

	this->PostMessage( WM_CALLBACK_STATUS, (WPARAM)hr );
}

HRESULT __stdcall LoadModelDlg::progressCallback( double val, void* pv ) noexcept
{
	LoadModelDlg& dialog = *(LoadModelDlg*)pv;
	constexpr double mul = progressMaxInteger;
	int pos = lround( mul * val );
	dialog.progressBar.PostMessage( PBM_SETPOS, pos, 0 );
	return S_OK;
}

LRESULT LoadModelDlg::OnCallbackStatus( UINT, WPARAM wParam, LPARAM, BOOL& bHandled )
{
	setPending( false );

	bHandled = TRUE;
	const HRESULT hr = (HRESULT)wParam;
	if( FAILED( hr ) )
	{
		LPCTSTR failMessage = L"Error loading the model";
		if( loadError.GetLength() > 0 )
		{
			CString tmp = failMessage;
			tmp += L"\n";
			tmp += loadError;
			return validationError( tmp, hr );
		}
		else
			return validationError( failMessage, hr );
	}

	appState.source.path = path;
	appState.source.impl = impl;
	appState.saveModelSource();

	EndDialog( IDOK );
	return 0;
}

LRESULT LoadModelDlg::OnHyperlink( int idCtrl, LPNMHDR pnmh, BOOL& bHandled )
{
	const UINT code = pnmh->code;
	switch( code )
	{
	case NM_CLICK:
	case NM_RETURN:
		break;
	default:
		return 0;
	}

	PNMLINK pNMLink = (PNMLINK)pnmh;
	LPCTSTR url = pNMLink->item.szUrl;
	ShellExecute( NULL, L"open", url, NULL, NULL, SW_SHOW );
	bHandled = TRUE;
	return 0;
}

void LoadModelDlg::onModelAdvanced()
{
	ModelAdvancedDlg dlg{ appState };
	dlg.show( m_hWnd );
}