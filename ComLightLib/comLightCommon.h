#pragma once
#include "hresult.h"

#ifdef _MSC_VER
#include <guiddef.h>
#else
#include "pal/guiddef.h"
using LPCTSTR = const char*;
#endif

#include "unknwn.h"