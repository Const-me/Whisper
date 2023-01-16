#pragma once
// Setup Windows SDK to only enable features available since Windows 8.0
#include <WinSDKVer.h>
#define _WIN32_WINNT _WIN32_WINNT_WIN8
#define NTDDI_VERSION NTDDI_WIN8
#include <SDKDDKVer.h>
