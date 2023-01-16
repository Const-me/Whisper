#pragma once
#include "AppState.h"
#include "Utils/WTL/atlddx.h"
#include "Utils/miscUtils.h"
#include "Utils/LanguageDropdown.h"
#include "Utils/TranslateCheckbox.h"
#include "Utils/PendingState.h"
#include "CircleIndicator.h"

class CaptureDlg :
	public CDialogImpl<CaptureDlg>,
	public CWinDataExchange<CaptureDlg>,
	public iThreadPoolCallback
{
	AppState& appState;

public:
	static constexpr UINT IDD = IDD_CAPTURE_DIALOG;
	static constexpr UINT WM_CALLBACK_COMPLETION = WM_APP + 1;
	static constexpr UINT WM_CALLBACK_STATUS = WM_APP + 2;

	CaptureDlg( AppState& app ) : appState( app ) { }

	HRESULT show();

	BEGIN_MSG_MAP( CaptureDlg )
		MESSAGE_HANDLER( WM_INITDIALOG, OnInitDialog )
		ON_BUTTON_CLICK( IDC_CONSOLE, cbConsole.click )
		ON_BUTTON_CLICK( IDC_DEV_REFRESH, onDeviceRefresh );
		ON_BUTTON_CLICK( IDC_BROWSE_RESULT, onBrowseResult );
		ON_BUTTON_CLICK( IDC_SAVE_TEXT, onSaveTextCheckbox );
		ON_BUTTON_CLICK( IDC_RUN_CAPTURE, onRunCapture );

		ON_BUTTON_CLICK( IDCANCEL, onClose )
		ON_BUTTON_CLICK( IDC_BACK, onBack )
		ON_BUTTON_CLICK( IDC_TRANSCRIBE, onTranscribe );

		MESSAGE_HANDLER( WM_CALLBACK_COMPLETION, onThreadQuit );
		MESSAGE_HANDLER( WM_CALLBACK_STATUS, onThreadStatus );
	END_MSG_MAP()

	BEGIN_DDX_MAP( CaptureDlg )
		DDX_CONTROL_HANDLE( IDC_DEVICE, cbCaptureDevice )
		DDX_CONTROL_HANDLE( IDC_RUN_CAPTURE, btnRunCapture );
		DDX_CONTROL_HANDLE( IDC_TRANSCRIBE_PROGRESS, progressBar );
		DDX_CONTROL_HANDLE( IDC_SAVE_TEXT, checkSave )
		DDX_CONTROL_HANDLE( IDC_SAVE_APPEND, checkAppend )
		DDX_CONTROL_HANDLE( IDC_SAVE_TIMESTAMPS, checkTimestamps )
		DDX_CONTROL_HANDLE( IDC_PATH_RESULT, transcribeOutputPath )
		DDX_CONTROL_HANDLE( IDC_BROWSE_RESULT, transcribeOutputBrowse );

		DDX_CONTROL( IDC_VOICE_ACTIVITY, voiceActivity );
		DDX_CONTROL( IDC_TRANS_STATUS, transcribeActivity );
		DDX_CONTROL( IDC_STALL_STATUS, stalled );
		
	END_DDX_MAP()

private:
	PendingState pendingState;
	void setPending( bool nowPending );

	LRESULT OnInitDialog( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled );

	void onClose()
	{
		ATLVERIFY( EndDialog( IDCANCEL ) );
	}
	void onBack()
	{
		ATLVERIFY( EndDialog( IDC_BACK ) );
	}
	void onTranscribe()
	{
		ATLVERIFY( EndDialog( IDC_TRANSCRIBE ) );
	}

	// List capture devices, and populate the combobox
	bool listDevices();
	void onDeviceRefresh();
	bool selectDevice( LPCTSTR endpoint );

	static HRESULT __stdcall listDevicesCallback( int len, const Whisper::sCaptureDevice* buffer, void* pv ) noexcept;
	ConsoleCheckbox cbConsole;
	LanguageDropdown languageSelector;
	TranslateCheckbox cbTranslate;
	CComboBox cbCaptureDevice;

	void onBrowseResult();

	enum struct eTextFlags : uint32_t;
	CButton	checkSave, checkAppend, checkTimestamps;
	CEdit transcribeOutputPath;
	CButton transcribeOutputBrowse;
	void onSaveTextCheckbox();
	eTextFlags getOutputFlags();

	CButton btnRunCapture;
	CProgressBarCtrl progressBar;
	ThreadPoolWork work;

	struct sCaptureDevice
	{
		CString displayName;
		CString endpoint;
	};
	std::vector<sCaptureDevice> devices;

	void showError( LPCTSTR text, HRESULT hr );

	CircleIndicator voiceActivity;
	CircleIndicator transcribeActivity;
	CircleIndicator stalled;

	struct sThreadState
	{
		volatile bool stopRequested;
		bool translate;
		eTextFlags flags;
		CAtlFile* file;
		uint32_t language;
		Whisper::sCaptureParams captureParams;
		CString endpoint;
		CString textOutputPath;
		CString errorMessage;
	};
	sThreadState threadState;
	bool captureRunning = false;

	void getThreadError();
	void onRunCapture();
	HRESULT runCapture();
	void __stdcall poolCallback() noexcept override final;

	LRESULT onThreadQuit( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled );
	LRESULT onThreadStatus( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled );

	static HRESULT __stdcall cbCancel( void* pv ) noexcept;
	static HRESULT __stdcall cbStatus( void* pv, Whisper::eCaptureStatus status ) noexcept;

	static HRESULT __cdecl newSegmentCallback( Whisper::iContext* ctx, uint32_t n_new, void* user_data ) noexcept;

	HRESULT appendTextFile( Whisper::iTranscribeResult* results, uint32_t newSegments );
};