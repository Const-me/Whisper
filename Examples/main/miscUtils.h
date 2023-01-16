#pragma once
#include <string>

std::string utf8( const std::wstring& utf16 );

std::wstring utf16( const std::string& u8 );

using HRESULT = long;
void printError( const char* what, HRESULT hr );