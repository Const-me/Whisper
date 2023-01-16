using System.Runtime.InteropServices;

/// <summary>Utility class to setup console coloring with ANSI codes.</summary>
/// <remarks>The feature requires Windows 10 or newer</remarks>
static class AnsiCodes
{
	const string dll = "kernel32.dll";

	[DllImport( dll, SetLastError = true )]
	static extern IntPtr GetStdHandle( int nStdHandle );

	const int STD_OUTPUT_HANDLE = -11;

	[Flags]
	enum ConsoleModes: uint
	{
		// Input flags
		ENABLE_PROCESSED_INPUT = 0x0001,
		ENABLE_LINE_INPUT = 0x0002,
		ENABLE_ECHO_INPUT = 0x0004,
		ENABLE_WINDOW_INPUT = 0x0008,
		ENABLE_MOUSE_INPUT = 0x0010,
		ENABLE_INSERT_MODE = 0x0020,
		ENABLE_QUICK_EDIT_MODE = 0x0040,
		ENABLE_EXTENDED_FLAGS = 0x0080,
		ENABLE_AUTO_POSITION = 0x0100,
		ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200,

		// Output flags
		ENABLE_PROCESSED_OUTPUT = 0x0001,
		ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002,
		ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004,
		DISABLE_NEWLINE_AUTO_RETURN = 0x0008,
		ENABLE_LVB_GRID_WORLDWIDE = 0x0010
	}

	[DllImport( dll, SetLastError = true )]
	static extern bool GetConsoleMode( IntPtr hConsoleHandle, out ConsoleModes mode );

	[DllImport( dll, SetLastError = true )]
	static extern bool SetConsoleMode( IntPtr hConsoleHandle, ConsoleModes mode );

	static AnsiCodes()
	{
		IntPtr h = GetStdHandle( STD_OUTPUT_HANDLE );
		IntPtr INVALID_HANDLE_VALUE = (IntPtr)( -1 );
		if( h == INVALID_HANDLE_VALUE )
			return;

		if( !GetConsoleMode( h, out ConsoleModes mode ) )
			return;

		if( mode.HasFlag( ConsoleModes.ENABLE_VIRTUAL_TERMINAL_PROCESSING ) )
		{
			enabled = true;
			return;
		}

		mode |= ConsoleModes.ENABLE_VIRTUAL_TERMINAL_PROCESSING;
		if( SetConsoleMode( h, mode ) )
		{
			enabled = true;
			return;
		}
	}

	public static readonly bool enabled = false;
}