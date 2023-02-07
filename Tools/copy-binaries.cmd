if exist ..\..\Release\Next rmdir /s /q ..\..\Release\Next
mkdir ..\..\Release\Next

rem Desktop example
mkdir ..\..\Release\Next\WhisperDesktop
copy ..\x64\Release\Whisper.dll ..\..\Release\Next\WhisperDesktop\
copy ..\x64\Release\WhisperDesktop.exe ..\..\Release\Next\WhisperDesktop\

rem Library
mkdir ..\..\Release\Next\Library
mkdir ..\..\Release\Next\Library\Binary
copy ..\x64\Release\Whisper.dll ..\..\Release\Next\Library\Binary\
mkdir ..\..\Release\Next\Library\Include
copy ..\Whisper\API\*.* ..\..\Release\Next\Library\Include\
mkdir ..\..\Release\Next\Library\Linker
copy ..\x64\Release\Whisper.lib ..\..\Release\Next\Library\Linker\

rem debug symbols
copy ..\x64\Release\Whisper.pdb ..\..\Release\Next\

rem C++ CLI example
mkdir ..\..\Release\Next\cli
copy ..\x64\Release\Whisper.dll ..\..\Release\Next\cli\
copy ..\x64\Release\main.exe ..\..\Release\Next\cli\