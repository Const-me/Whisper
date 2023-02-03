#include "stdafx.h"
#include "CaptureDlg.h"

HRESULT CaptureDlg::show()
{
	auto res = DoModal( nullptr );
	if( res == -1 )
		return HRESULT_FROM_WIN32( GetLastError() );
	switch( res )
	{
	case IDC_BACK:
		return SCREEN_MODEL;
	case IDC_TRANSCRIBE:
		return SCREEN_TRANSCRIBE;
	}
	return S_OK;
}

static const LPCTSTR regValDevice = L"captureDevice";
static const LPCTSTR regValOutPath = L"captureTextFile";
static const LPCTSTR regValOutFormat = L"captureTextFlags";

enum struct CaptureDlg::eTextFlags : uint32_t
{
	Save = 1,
	Append = 2,
	Timestamps = 4,
};

LRESULT CaptureDlg::OnInitDialog( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled )
{
	// First DDX call, hooks up variables to controls.
	DoDataExchange( false );

	languageSelector.initialize( m_hWnd, IDC_LANGUAGE, appState );
	cbTranslate.initialize( m_hWnd, IDC_TRANSLATE, appState );
	cbConsole.initialize( m_hWnd, IDC_CONSOLE, appState );

	pendingState.initialize(
		// Controls to disable while pending, re-enable afterwards
		{
			languageSelector, GetDlgItem( IDC_TRANSLATE ),
			cbCaptureDevice,
			checkSave, checkAppend, checkTimestamps, transcribeOutputPath, transcribeOutputBrowse,
			GetDlgItem( IDC_DEV_REFRESH ),
			GetDlgItem( IDC_BACK ),
			GetDlgItem( IDC_TRANSCRIBE ),
			GetDlgItem( IDCANCEL ),
		},
		// Controls to show while pending, hide afterwards
		{
			voiceActivity, GetDlgItem( IDC_VOICE_ACTIVITY_LBL ),
			transcribeActivity, GetDlgItem( IDC_TRANS_LBL ),
			stalled, GetDlgItem( IDC_STALL_LBL ),
			progressBar,
		} );

	stalled.setActiveColor( flipRgb( 0xffcc33 ) );

	HRESULT hr = work.create( this );
	if( FAILED( hr ) )
	{
		reportError( m_hWnd, L"CreateThreadpoolWork failed", nullptr, hr );
		EndDialog( IDCANCEL );
	}

	listDevices();
	selectDevice( appState.stringLoad( regValDevice ) );

	constexpr uint32_t defaultFlags = (uint32_t)eTextFlags::Append;
	uint32_t flags = appState.dwordLoad( regValOutFormat, defaultFlags );
	if( flags & (uint32_t)eTextFlags::Save )
		checkSave.SetCheck( BST_CHECKED );
	if( flags & (uint32_t)eTextFlags::Append )
		checkAppend.SetCheck( BST_CHECKED );
	if( flags & (uint32_t)eTextFlags::Timestamps )
		checkTimestamps.SetCheck( BST_CHECKED );

	transcribeOutputPath.SetWindowText( appState.stringLoad( regValOutPath ) );
	onSaveTextCheckbox();

	appState.lastScreenSave( SCREEN_CAPTURE );
	appState.setupIcon( this );
	ATLVERIFY( CenterWindow() );
	return 0;
}

HRESULT __stdcall CaptureDlg::listDevicesCallback( int len, const Whisper::sCaptureDevice* buffer, void* pv ) noexcept
{
	std::vector<sCaptureDevice>& devices = *( std::vector<sCaptureDevice> * )pv;
	devices.resize( len );
	for( int i = 0; i < len; i++ )
	{
		devices[ i ].displayName = buffer[ i ].displayName;
		devices[ i ].endpoint = buffer[ i ].endpoint;
	}
	return S_OK;
}

bool CaptureDlg::listDevices()
{
	appState.mediaFoundation->listCaptureDevices( &listDevicesCallback, &devices );
	cbCaptureDevice.ResetContent();
	for( const auto& dev : devices )
		cbCaptureDevice.AddString( dev.displayName );
	return !devices.empty();
}

void CaptureDlg::onDeviceRefresh()
{
	// Save the current selection
	const int curSel = cbCaptureDevice.GetCurSel();
	CString str;
	if( curSel >= 0 && curSel < (int)devices.size() )
		str = std::move( devices[ curSel ].endpoint );

	// Refresh
	listDevices();

	// Restore the selection
	selectDevice( str );

	const size_t len = devices.size();
	if( len == 0 )
	{
		MessageBox( L"No capture devices found on this computer.\nIf you have a USB microphone, connect it to this PC,\nand press “refresh” button.",
			L"Capture Devices", MB_OK | MB_ICONWARNING );
	}
	else
	{
		const char* suffix = ( len != 1 ) ? "s" : "";
		str.Format( L"Detected %zu audio capture device%S.", len, suffix );
		MessageBox( str, L"Capture Devices", MB_OK | MB_ICONINFORMATION );
	}
}

bool CaptureDlg::selectDevice( LPCTSTR endpoint )
{
	if( nullptr != endpoint && 0 != *endpoint )
	{
		for( size_t i = 0; i < devices.size(); i++ )
		{
			if( devices[ i ].endpoint == endpoint )
			{
				cbCaptureDevice.SetCurSel( (int)i );
				return true;
			}
		}
	}

	if( !devices.empty() )
		cbCaptureDevice.SetCurSel( 0 );
	return false;
}

void CaptureDlg::onSaveTextCheckbox()
{
	const BOOL enabled = ( checkSave.GetCheck() == BST_CHECKED );
	std::array<HWND, 4> controls = { checkAppend, checkTimestamps, transcribeOutputPath, transcribeOutputBrowse };
	for( HWND w : controls )
		::EnableWindow( w, enabled );
}

void CaptureDlg::onBrowseResult()
{
	LPCTSTR title = L"Output Text File";
	LPCTSTR outputFilters = L"Text files (*.txt)\0*.txt\0\0";
	CString path;
	transcribeOutputPath.GetWindowText( path );
	if( !getSaveFileName( m_hWnd, title, outputFilters, path ) )
		return;

	LPCTSTR ext = PathFindExtension( path );
	if( 0 == *ext )
	{
		wchar_t* const buffer = path.GetBufferSetLength( path.GetLength() + 5 );
		PathRenameExtension( buffer, L".txt" );
		path.ReleaseBuffer();
	}

	transcribeOutputPath.SetWindowText( path );
}

CaptureDlg::eTextFlags CaptureDlg::getOutputFlags()
{
	uint32_t flags = 0;
	if( checkSave.GetCheck() == BST_CHECKED )
		flags |= (uint32_t)eTextFlags::Save;
	if( checkAppend.GetCheck() == BST_CHECKED )
		flags |= (uint32_t)eTextFlags::Append;
	if( checkTimestamps.GetCheck() == BST_CHECKED )
		flags |= (uint32_t)eTextFlags::Timestamps;
	return (eTextFlags)flags;
}

void CaptureDlg::setPending( bool nowPending )
{
	pendingState.setPending( nowPending );
	if( nowPending )
	{
		progressBar.SetMarquee( TRUE, 0 );
		btnRunCapture.SetWindowText( L"Stop" );
	}
	else
	{
		progressBar.SetMarquee( FALSE, 0 );
		btnRunCapture.SetWindowText( L"Capture" );
		btnRunCapture.EnableWindow( TRUE );
		captureRunning = false;
	}
}

void CaptureDlg::onRunCapture()
{
	if( captureRunning )
	{
		threadState.stopRequested = true;
		btnRunCapture.EnableWindow( FALSE );
		return;
	}

	int dev = cbCaptureDevice.GetCurSel();
	if( dev < 0 || dev >= (int)devices.size() )
	{
		showError( L"Please select a capture device", S_FALSE );
		return;
	}
	threadState.endpoint = devices[ dev ].endpoint;
	threadState.language = languageSelector.selectedLanguage();
	threadState.translate = cbTranslate.checked();
	if( isInvalidTranslate( m_hWnd, threadState.language, threadState.translate ) )
		return;

	threadState.flags = getOutputFlags();
	if( (uint32_t)threadState.flags & (uint32_t)eTextFlags::Save )
	{
		transcribeOutputPath.GetWindowText( threadState.textOutputPath );
		if( threadState.textOutputPath.GetLength() <= 0 )
		{
			showError( L"Please specify the output text file", S_FALSE );
			return;
		}
		appState.stringStore( regValOutPath, threadState.textOutputPath );
	}
	else
		cbConsole.ensureChecked();

	languageSelector.saveSelection( appState );
	cbTranslate.saveSelection( appState );
	appState.stringStore( regValDevice, threadState.endpoint );
	appState.dwordStore( regValOutFormat, (uint32_t)threadState.flags );

	captureRunning = true;
	threadState.errorMessage = L"";
	threadState.stopRequested = false;
	threadState.captureParams.minDuration = 7;
	threadState.captureParams.maxDuration = 11;
	setPending( true );
	work.post();
}

void __declspec( noinline ) CaptureDlg::getThreadError()
{
	getLastError( threadState.errorMessage );
}

#define CHECK_EX( hr ) { const HRESULT __hr = ( hr ); if( FAILED( __hr ) ) { getThreadError(); return __hr; } }

static HRESULT appendDate( CString& str, const SYSTEMTIME& time )
{
	constexpr DWORD dateFlags = DATE_LONGDATE;
	int cc = GetDateFormatEx( LOCALE_NAME_USER_DEFAULT, dateFlags, &time, nullptr, nullptr, 0, nullptr );
	if( 0 == cc )
		return getLastHr();

	const int oldLength = str.GetLength();
	wchar_t* const buffer = str.GetBufferSetLength( oldLength + cc );
	cc = GetDateFormatEx( LOCALE_NAME_USER_DEFAULT, dateFlags, &time, nullptr, buffer + oldLength, cc, nullptr );
	if( 0 != cc )
	{
		str.ReleaseBuffer();
		return S_OK;
	}
	HRESULT hr = getLastHr();
	str.ReleaseBuffer();
	return hr;
}

static HRESULT appendTime( CString& str, const SYSTEMTIME& time )
{
	constexpr DWORD timeFlags = 0;
	int cc = GetTimeFormatEx( LOCALE_NAME_USER_DEFAULT, timeFlags, &time, nullptr, nullptr, 0 );
	if( 0 == cc )
		return getLastHr();

	const int oldLength = str.GetLength();
	wchar_t* const buffer = str.GetBufferSetLength( oldLength + cc );
	cc = GetTimeFormatEx( LOCALE_NAME_USER_DEFAULT, timeFlags, &time, nullptr, buffer + oldLength, cc );
	if( 0 != cc )
	{
		str.ReleaseBuffer();
		return S_OK;
	}
	HRESULT hr = getLastHr();
	str.ReleaseBuffer();
	return hr;
}

static HRESULT printDateTime( CAtlFile& file )
{
	SYSTEMTIME time;
	GetLocalTime( &time );

	CString str;
	str = L"==== Captured on ";
	CHECK( appendDate( str, time ) );
	str += L", ";
	CHECK( appendTime( str, time ) );
	str += L" ====\r\n";

	CStringA u8;
	makeUtf8( u8, str );
	return file.Write( cstr( u8 ), (DWORD)u8.GetLength() );
}

inline HRESULT CaptureDlg::runCapture()
{
	clearLastError();
	using namespace Whisper;
	CComPtr<iAudioCapture> capture;
	CHECK_EX( appState.mediaFoundation->openCaptureDevice( threadState.endpoint, threadState.captureParams, &capture ) );

	HRESULT hr;
	CAtlFile file;
	const uint32_t flags = (uint32_t)threadState.flags;
	if( flags & (uint32_t)eTextFlags::Save )
	{
		const bool append = 0 != ( flags & (uint32_t)eTextFlags::Append );
		const DWORD creation = append ? OPEN_ALWAYS : CREATE_ALWAYS;
		hr = file.Create( threadState.textOutputPath, GENERIC_WRITE, FILE_SHARE_READ, creation );
		if( FAILED( hr ) )
		{
			threadState.errorMessage = L"Unable to create the output text file";
			return hr;
		}
		if( append )
		{
			ULONGLONG size;
			CHECK( file.GetSize( size ) );
			if( size == 0 )
				CHECK( writeUtf8Bom( file ) )
			else
				CHECK( file.Seek( 0, SEEK_END ) );
		}
		else
		{
			CHECK( writeUtf8Bom( file ) );
		}

		if( flags & (uint32_t)eTextFlags::Timestamps )
			CHECK( printDateTime( file ) );

		threadState.file = &file;
	}
	else
		threadState.file = nullptr;

	CComPtr<iContext> context;
	CHECK_EX( appState.model->createContext( &context ) );

	sFullParams fullParams;
	CHECK_EX( context->fullDefaultParams( eSamplingStrategy::Greedy, &fullParams ) );
	fullParams.language = threadState.language;
	fullParams.setFlag( eFullParamsFlags::Translate, threadState.translate );
	fullParams.resetFlag( eFullParamsFlags::PrintRealtime );
	fullParams.new_segment_callback = &newSegmentCallback;
	fullParams.new_segment_callback_user_data = this;

	sCaptureCallbacks callbacks;
	callbacks.shouldCancel = &cbCancel;
	callbacks.captureStatus = &cbStatus;
	callbacks.pv = this;

	CHECK_EX( context->runCapture( fullParams, callbacks, capture ) );
	threadState.file = nullptr;

	context->timingsPrint();
	return S_OK;
}

void __stdcall CaptureDlg::poolCallback() noexcept
{
	const HRESULT hr = runCapture();
	PostMessage( WM_CALLBACK_COMPLETION, hr );
}

void CaptureDlg::showError( LPCTSTR text, HRESULT hr )
{
	reportError( m_hWnd, text, L"Capture failed", hr );
}

LRESULT CaptureDlg::onThreadQuit( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled )
{
	setPending( false );

	const HRESULT hr = (HRESULT)wParam;
	if( FAILED( hr ) )
	{
		LPCTSTR failMessage = L"Capture failed";

		if( threadState.errorMessage.GetLength() > 0 )
		{
			CString tmp = failMessage;
			tmp += L"\n";
			tmp += threadState.errorMessage;
			showError( tmp, hr );
		}
		else
			showError( failMessage, hr );

		return 0;
	}
	else
	{
		if( (uint32_t)threadState.flags & (uint32_t)eTextFlags::Save )
			ShellExecute( NULL, L"open", threadState.textOutputPath, NULL, NULL, SW_SHOW );
	}

	return 0;
}

LRESULT CaptureDlg::onThreadStatus( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled )
{
	using namespace Whisper;
	const uint8_t newStatus = (uint8_t)wParam;
	// Update the GUI
	voiceActivity.setActive( 0 != ( newStatus & (uint8_t)eCaptureStatus::Voice ) );
	transcribeActivity.setActive( 0 != ( newStatus & (uint8_t)eCaptureStatus::Transcribing ) );
	stalled.setActive( 0 != ( newStatus & (uint8_t)eCaptureStatus::Stalled ) );
	return 0;
}

HRESULT __stdcall CaptureDlg::cbCancel( void* pv ) noexcept
{
	const bool stopRequested = ( (CaptureDlg*)pv )->threadState.stopRequested;
	return stopRequested ? S_FALSE : S_OK;
}

HRESULT __stdcall CaptureDlg::cbStatus( void* pv, Whisper::eCaptureStatus status ) noexcept
{
	CaptureDlg& dialog = *(CaptureDlg*)pv;
	if( dialog.PostMessage( WM_CALLBACK_STATUS, (uint8_t)status ) )
		return S_OK;
	return getLastHr();
}

HRESULT __cdecl CaptureDlg::newSegmentCallback( Whisper::iContext* ctx, uint32_t n_new, void* user_data ) noexcept
{
	using namespace Whisper;
	CComPtr<iTranscribeResult> result;
	const eResultFlags flags = eResultFlags::Timestamps | eResultFlags::Tokens;
	CHECK( ctx->getResults( flags, &result ) );
	CHECK( logNewSegments( result, n_new ) );

	CaptureDlg& dialog = *(CaptureDlg*)user_data;
	return dialog.appendTextFile( result, n_new );
}

HRESULT CaptureDlg::appendTextFile( Whisper::iTranscribeResult* results, uint32_t newSegments )
{
	if( nullptr == threadState.file || 0 == newSegments )
		return S_OK;

	using namespace Whisper;
	sTranscribeLength length;
	CHECK( results->getSize( length ) );

	const size_t len = length.countSegments;
	size_t i = len - newSegments;

	const sSegment* const segments = results->getSegments();
	CStringA str;
	for( ; i < len; i++ )
	{
		const sSegment& seg = segments[ i ];
		if( 0 != ( (uint32_t)threadState.flags & (uint32_t)eTextFlags::Timestamps ) )
		{
			str = "[";
			printTime( str, seg.time.begin );
			str += " --> ";
			printTime( str, seg.time.end );
			str += "]  ";
		}
		else
			str = "";

		str += seg.text;
		str += "\r\n";

		CHECK( threadState.file->Write( cstr( str ), (DWORD)str.GetLength() ) );
	}

	CHECK( threadState.file->Flush() );
	return S_OK;
}