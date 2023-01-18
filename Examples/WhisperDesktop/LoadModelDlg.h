#pragma once
#include "AppState.h"
#include "Utils/WTL/atlddx.h"
#include "Utils/miscUtils.h"

class LoadModelDlg:
	public CDialogImpl<LoadModelDlg>,
	public CWinDataExchange<LoadModelDlg>,
	public iThreadPoolCallback
{
	AppState& appState;
public:
	static constexpr UINT IDD = IDD_OPEN_MODEL;
	static constexpr UINT WM_CALLBACK_STATUS = WM_APP + 1;

	LoadModelDlg( AppState& app ) : appState( app ) { }

	HRESULT show();

	BEGIN_MSG_MAP( LoadModelDlg )
		MESSAGE_HANDLER( WM_INITDIALOG, OnInitDialog )
		ON_BUTTON_CLICK( IDC_CONSOLE, cbConsole.click )
		COMMAND_ID_HANDLER( IDCANCEL, OnCommand )
		COMMAND_ID_HANDLER( IDOK, OnOk )
		COMMAND_ID_HANDLER( IDC_BROWSE, OnBrowse )
		MESSAGE_HANDLER( WM_CALLBACK_STATUS, OnCallbackStatus )
		NOTIFY_ID_HANDLER( IDC_HYPERLINKS, OnHyperlink )
		ON_BUTTON_CLICK( IDC_MODEL_ADV, onModelAdvanced )
	END_MSG_MAP()

	BEGIN_DDX_MAP( LoadModelDlg )
		DDX_CONTROL_HANDLE( IDC_PATH, modelPath )
		DDX_CONTROL_HANDLE( IDC_MODEL_TYPE, cbModelType )
		DDX_CONTROL_HANDLE( IDC_PROGRESS, progressBar );
	END_DDX_MAP()
	
private:
	std::vector<HWND> editorsWindows;
	std::vector<HWND> pendingWindows;
	void setPending( bool nowPending );

	LRESULT OnInitDialog( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled );
	LRESULT OnCallbackStatus( UINT, WPARAM wParam, LPARAM, BOOL& bHandled );

	LRESULT OnCommand( UINT, INT nIdentifier, HWND, BOOL& bHandled )
	{
		ATLVERIFY( EndDialog( nIdentifier ) );
		return 0;
	}
	LRESULT OnBrowse( UINT, INT, HWND, BOOL& bHandled );
	LRESULT OnOk( UINT, INT, HWND, BOOL& bHandled );

	ConsoleCheckbox cbConsole;
	CComboBox cbModelType;
	CEdit modelPath;
	CProgressBarCtrl progressBar;

	LRESULT validationError( LPCTSTR message );
	LRESULT validationError( LPCTSTR message, HRESULT hr );

	ThreadPoolWork work;
	CString path;
	Whisper::eModelImplementation impl;
	CString loadError;
	void __stdcall poolCallback() noexcept override final;

	LRESULT OnHyperlink( int idCtrl, LPNMHDR pnmh, BOOL& bHandled );

	static HRESULT __stdcall progressCallback( double val, void* pv ) noexcept;

	void onModelAdvanced();
};