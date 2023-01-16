#include "stdafx.h"
#include "miscUtils.h"

namespace
{
	wchar_t* formatMessage( HRESULT hr )
	{
		wchar_t* err;
		if( FormatMessage( FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
			NULL,
			hr,
			MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ),
			(LPTSTR)&err,
			0,
			NULL ) )
			return err;
		return nullptr;
	}
}

CString formatErrorMessage( HRESULT hr )
{
	CString message;
	const wchar_t* err = formatMessage( hr );
	if( nullptr != err )
	{
		message = err;
		LocalFree( (HLOCAL)err );
		message.TrimRight();
	}
	else
		message.Format( L"Error code %i (0x%08X)", hr, hr );

	return message;
}

void reportFatalError( const char* what, HRESULT hr )
{
	CString message;
	message.Format( L"%S\n%S\n", "Unable to start the application.", what );
	message += formatErrorMessage( hr );
	::MessageBox( nullptr, message, L"Whisper Desktop Startup", MB_OK | MB_ICONERROR );
}

namespace
{
	using Whisper::eModelImplementation;

	struct sImplString
	{
		eModelImplementation val;
		LPCTSTR str;
	};
	static const std::array<sImplString, 3> s_implStrings =
	{
		sImplString{ eModelImplementation::GPU, L"GPU" },
		sImplString{ eModelImplementation::Hybrid, L"Hybrid" },
		sImplString{ eModelImplementation::Reference, L"Reference" },
	};
}

HRESULT implParse( const CString& s, eModelImplementation& rdi )
{
	for( const auto& is : s_implStrings )
	{
		if( 0 != s.CompareNoCase( is.str ) )
			continue;
		rdi = is.val;;
		return S_OK;
	}
	return E_INVALIDARG;
}

LPCTSTR implString( eModelImplementation i )
{
	for( const auto& is : s_implStrings )
		if( is.val == i )
			return is.str;
	return nullptr;
}

void implPopulateCombobox( CComboBox& cb, Whisper::eModelImplementation impl )
{
	int curSel = 0;
	int idx = 0;
	for( const auto& is : s_implStrings )
	{
		cb.AddString( is.str );
		if( is.val == impl )
			curSel = idx;
		idx++;
	}
	cb.SetCurSel( curSel );
}

Whisper::eModelImplementation implGetValue( CComboBox& cb )
{
	int curSel = cb.GetCurSel();
	if( curSel < 0 )
		return (Whisper::eModelImplementation)0;
	return s_implStrings[ curSel ].val;
}

ThreadPoolWork::~ThreadPoolWork()
{
	if( nullptr != work )
	{
		CloseThreadpoolWork( work );
		work = nullptr;
	}
}

void __stdcall ThreadPoolWork::callback( PTP_CALLBACK_INSTANCE Instance, PVOID Context, PTP_WORK Work )
{
	iThreadPoolCallback* cb = (iThreadPoolCallback*)Context;
	cb->poolCallback();
}

HRESULT ThreadPoolWork::create( iThreadPoolCallback* cb )
{
	if( nullptr == cb )
		return E_POINTER;
	if( nullptr != work )
		return HRESULT_FROM_WIN32( ERROR_ALREADY_INITIALIZED );

	work = CreateThreadpoolWork( &callback, cb, nullptr );
	if( nullptr != work )
		return S_OK;

	return HRESULT_FROM_WIN32( GetLastError() );
}

HRESULT ThreadPoolWork::post()
{
	if( nullptr == work )
		return OLE_E_BLANK;
	SubmitThreadpoolWork( work );
	return S_OK;
}

void makeUtf16( CString& rdi, const char* utf8 )
{
	const size_t length = strlen( utf8 );
	int count = MultiByteToWideChar( CP_UTF8, 0, utf8, (int)length, nullptr, 0 );
	wchar_t* p = rdi.GetBufferSetLength( count );
	MultiByteToWideChar( CP_UTF8, 0, utf8, (int)length, p, count );
	rdi.ReleaseBuffer();
}

void makeUtf8( CStringA& rdi, const CString& utf16 )
{
	int count = WideCharToMultiByte( CP_UTF8, 0, utf16, utf16.GetLength(), nullptr, 0, nullptr, nullptr );
	char* s = rdi.GetBufferSetLength( count + 1 );
	count = WideCharToMultiByte( CP_UTF8, 0, utf16, utf16.GetLength(), s, count, nullptr, nullptr );
	rdi.ReleaseBufferSetLength( count );
}

constexpr int ofnBufferLength = 2048;

bool getOpenFileName( HWND owner, LPCTSTR title, LPCTSTR filter, CString& path )
{
	wchar_t buffer[ ofnBufferLength ];
	buffer[ 0 ] = 0;
	OPENFILENAME ofn;
	memset( &ofn, 0, sizeof( ofn ) );
	ofn.lStructSize = sizeof( OPENFILENAME );
	ofn.hwndOwner = owner;
	ofn.lpstrFilter = filter;
	ofn.lpstrTitle = title;
	ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
	ofn.lpstrFile = buffer;
	ofn.nMaxFile = ofnBufferLength - 1;

	CString dir;
	if( path.GetLength() > 0 && path.GetLength() < ofnBufferLength )
		wcsncpy_s( buffer, path, path.GetLength() );

	if( !GetOpenFileName( &ofn ) )
	{
		path = L"";
		return false;
	}
	else
	{
		path = ofn.lpstrFile;
		return true;
	}
}

bool getSaveFileName( HWND owner, LPCTSTR title, LPCTSTR filter, CString& path, DWORD* filterIndex )
{
	wchar_t buffer[ ofnBufferLength ];
	buffer[ 0 ] = 0;

	OPENFILENAME ofn;
	memset( &ofn, 0, sizeof( ofn ) );
	ofn.lStructSize = sizeof( OPENFILENAME );
	ofn.hwndOwner = owner;
	ofn.lpstrFilter = filter;
	ofn.lpstrTitle = title;
	ofn.Flags = OFN_EXPLORER | OFN_PATHMUSTEXIST;
	ofn.lpstrFile = buffer;
	ofn.nMaxFile = ofnBufferLength - 1;
	if( nullptr != filterIndex )
		ofn.nFilterIndex = *filterIndex + 1;

	if( path.GetLength() > 0 && path.GetLength() < ofnBufferLength )
		wcsncpy_s( buffer, path, path.GetLength() );

	if( !GetSaveFileName( &ofn ) )
		return false;

	path = ofn.lpstrFile;

	if( nullptr != filterIndex )
		*filterIndex = ofn.nFilterIndex - 1;

	return true;
}

void reportError( HWND owner, LPCTSTR text, LPCTSTR title, HRESULT hr )
{
	if( nullptr == title )
		title = L"Operation Failed";

	CString message = text;
	message.TrimRight();
	if( FAILED( hr ) )
	{
		message += L"\n";
		message += formatErrorMessage( hr );
	}

	::MessageBox( owner, message, title, MB_OK | MB_ICONWARNING );
}

HRESULT writeUtf8Bom( CAtlFile& file )
{
	const std::array<uint8_t, 3> bom = { 0xEF, 0xBB, 0xBF };
	return file.Write( bom.data(), 3 );
}

bool isInvalidTranslate( HWND owner, uint32_t lang, bool translate )
{
	if( !translate )
		return false;
	constexpr uint32_t english = 0x6E65;
	if( lang != english )
		return false;

	LPCTSTR message = L"The translate feature translates speech to English.\nIt’s not available when the audio language is already English.";
	MessageBox( owner, message, L"Incompatible parameters", MB_OK | MB_ICONINFORMATION );
	return true;
}