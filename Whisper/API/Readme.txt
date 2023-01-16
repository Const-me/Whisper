The headers in this folder define the complete public API of Whisper.dll.

To consume the library in your C++ software, include exactly one of the following headers.

1. If you’re building a windows app, include whisperWindows.h header, and you'll get traditional Win32 COM projection of the API.

2. If you’re porting to other OS, or porting to different C++ compiler, or already using ComLight support library, include whisperComLight.h header.
If you do that, in addition to this "Whisper/API" folder you also gonna need the "ComLightLib" dependency.
This will get you the ComLight flavor of these COM interfaces.

Internally, the actual implementation uses the ComLight flavour of the interfaces, but that’s fine because they are binary compatible.

The reason for the difference between these flavors — Visual Studio’s CComPtr<T> and other related utilities expect interface IDs specified with __declspec(uuid) directive.

That language extension is specific to Visual C++, not supported in GCC nor Clang compilers.