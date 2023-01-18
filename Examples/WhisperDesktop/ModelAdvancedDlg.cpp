#include "stdafx.h"
#include "ModelAdvancedDlg.h"
using Whisper::eGpuModelFlags;

LRESULT ModelAdvancedDlg::onInitDialog( UINT nMessage, WPARAM wParam, LPARAM lParam, BOOL& bHandled )
{
	cbWave = GetDlgItem( IDC_WAVE );
	cbReshapedMatMul = GetDlgItem( IDC_RESHAPED_MAT_MUL );
	const uint32_t flags = appState.gpuFlagsLoad();

	// Setup the "Compute shaders" combobox
	cbWave.AddString( L"Wave64 shaders on AMD" );
	cbWave.AddString( L"Wave32, always" );
	cbWave.AddString( L"Wave64, always" );
	int i = 0;
	if( 0 != ( flags & (uint32_t)eGpuModelFlags::Wave32 ) )
		i = 1;
	else if( 0 != ( flags & (uint32_t)eGpuModelFlags::Wave64 ) )
		i = 2;
	cbWave.SetCurSel( i );

	// Setup the "Reshaped multiply" combobox
	cbReshapedMatMul.AddString( L"Reshape on AMD" );
	cbReshapedMatMul.AddString( L"Don’t reshape tensors" );
	cbReshapedMatMul.AddString( L"Reshape some tensors" );
	i = 0;
	if( 0 != ( flags & (uint32_t)eGpuModelFlags::NoReshapedMatMul ) )
		i = 1;
	else if( 0 != ( flags & (uint32_t)eGpuModelFlags::UseReshapedMatMul ) )
		i = 2;
	cbReshapedMatMul.SetCurSel( i );

	return 0;
}

bool ModelAdvancedDlg::show( HWND owner )
{
	auto res = DoModal( owner );
	return res == IDOK;
}

void ModelAdvancedDlg::onOk()
{
	// Gather values from these comboboxes
	uint32_t flags = 0;

	int i = cbWave.GetCurSel();
	if( 1 == i )
		flags |= (uint32_t)eGpuModelFlags::Wave32;
	else if( 2 == i )
		flags |= (uint32_t)eGpuModelFlags::Wave64;

	i = cbReshapedMatMul.GetCurSel();
	if( 1 == i )
		flags |= (uint32_t)eGpuModelFlags::NoReshapedMatMul;
	else if( 2 == i )
		flags |= (uint32_t)eGpuModelFlags::UseReshapedMatMul;

	// Save to registry
	appState.gpuFlagsStore( flags );

	EndDialog( IDOK );
}