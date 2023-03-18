The project in this folder compiles PowerShell wrapper for Whisper.

I consider it incomplete, but it works on my computer, you can try if you want.<br />
The supported PowerShell version is the one I have in my Windows 10, specifically it’s version 5.1

Usage example:

```
Import-Module C:\Temp\WhisperPS\WhisperPS.dll
$m = Import-WhisperModel D:\Data\Whisper\ggml-medium.bin
cd C:\Temp\2remove\Whisper
$res = dir .\* -include *.wma, *.wav | Transcribe-File $m
foreach ( $t in $res ) { $r = $t.SourceName + ".txt"; $t | Export-Text $r; }
foreach ( $t in $res ) { $r = $t.SourceName + ".ts.txt"; $t | Export-Text $r -timestamps; }
```

Unfortunately, PS 5.1 uses the legacy .NET framework 4.8.<br />
That’s why I couldn’t simply consume WhisperNet library.
Instead I implemented slightly different COM wrappers for the same C++ DLL.

Luckily, my [ComLightInterop](https://www.nuget.org/packages/ComLightInterop/) library is old,
and it still supports legacy .NET framework, along with other older versions of the runtime like .NET Core 2.1.