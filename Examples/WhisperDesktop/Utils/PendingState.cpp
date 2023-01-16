#include "stdafx.h"
#include "PendingState.h"

void PendingState::initialize( std::initializer_list<HWND> editors, std::initializer_list<HWND> pending )
{
	editorsWindows = editors;
	wasEnabled.resize( editorsWindows.size() );
	pendingWindows = pending;
}

void PendingState::setPending( bool nowPending )
{
	if( nowPending )
	{
		for( size_t i = 0; i < editorsWindows.size(); i++ )
		{
			BOOL e = IsWindowEnabled( editorsWindows[ i ] );
			if( e )
			{
				wasEnabled[ i ] = true;
				EnableWindow( editorsWindows[ i ], FALSE );
			}
			else
				wasEnabled[ i ] = false;
		}
	}
	else
	{
		for( size_t i = 0; i < editorsWindows.size(); i++ )
		{
			if( !wasEnabled[ i ] )
				continue;
			EnableWindow( editorsWindows[ i ], TRUE );
		}
	}

	const int show = nowPending ? SW_NORMAL : SW_HIDE;
	for( HWND w : pendingWindows )
		::ShowWindow( w, show );
}