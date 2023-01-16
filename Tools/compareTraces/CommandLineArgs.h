#pragma once

struct CommandLineArgs
{
	int64_t printDiff = -1;
	std::array<CString, 2> inputs;

	bool parse( int argc, wchar_t* argv[] );
};