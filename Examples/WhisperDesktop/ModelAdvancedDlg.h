#pragma once
#include "AppState.h"
#include "Utils/WTL/atlddx.h"
#include "Utils/miscUtils.h"

class ModelAdvancedDlg :
	public CDialogImpl<ModelAdvancedDlg>
{
	CComboBox cbWave, cbReshapedMatMul, cbAdapter;
	AppState& appState;

public:
	static constexpr UINT IDD = IDD_MODEL_ADV;

	ModelAdvancedDlg( AppState& app ) : appState( app ) { }

	BEGIN_MSG_MAP( ModelAdvancedDlg )
		MESSAGE_HANDLER( WM_INITDIALOG, onInitDialog )
		ON_BUTTON_CLICK( IDOK, onOk )
		ON_BUTTON_CLICK( IDCANCEL, onCancel )
	END_MSG_MAP()

	bool show( HWND owner );

private:

	LRESULT onInitDialog( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled );

	void onOk();

	void onCancel()
	{
		EndDialog( IDCANCEL );
	}
};