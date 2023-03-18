The `WhisperPS.csproj` project in this folder builds PowerShell wrapper for Whisper.

I wouldn’t call the wrapper particularly great, but it works on my computer.<br/>
This should handle use cases like “transcribe all files in a directory” or “export multiple formats”.

The supported PowerShell version is 5.1, the one I have preinstalled in my Windows 10 computer.<br/>
I wouldn’t expect it to work with the newer PowerShell Core, the runtime is different.

Usage example, assuming you’ve extracted WhisperPS.zip into C:\Temp\

```
Import-Module C:\Temp\WhisperPS\WhisperPS.dll -DisableNameChecking
$Model = Import-WhisperModel D:\Data\Whisper\ggml-medium.bin
cd C:\Temp\2remove\Whisper
$Results = dir .\* -include *.wma, *.wav | Transcribe-File $Model
foreach ( $i in $Results ) { $txt = $i.SourceName + ".txt"; $i | Export-Text $txt; }
foreach ( $i in $Results ) { $txt = $i.SourceName + ".ts.txt"; $i | Export-Text $txt -timestamps; }
```

If you’ve extracted to the local module directory, which is<br/>
`%USERPROFILE%\Documents\WindowsPowerShell\Modules`<br/>
you can simplify the first line, use the following one instead:<br/>
`Import-Module WhisperPS -DisableNameChecking`

Here’s the list of commands implemented by this module.

* `Get-Adapters` prints names of the graphics adapters visible to DirectCompute.<br/>
You can use these strings for the `-adapter` argument of the `Import-WhisperModel` command.
* `Import-WhisperModel` loads the model from disk, returns the object which keeps the model
* `Transcribe-File` loads audio file from disk, and transcribes the audio into text.
It returns the object which keeps both source file name, and transcribed text.
* `Export-Text` saves transcribed text into a text file, with or without timestamps.
* `Export-SubRip` saves transcribed text in *.srt format.
* `Export-WebVTT` saves transcribed text in *.vtt format.

By default, PowerShell doesn’t print any informational or debug messages.
If you want these messages, run these commends:

```
$InformationPreference="Continue"
$DebugPreference="Continue"
```

[Apparently](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_preference_variables?view=powershell-5.1),
the default value for these preference variables is `SilentlyContinue` so by default the messages go nowhere.

Unfortunately, PowerShell 5.1 uses legacy .NET framework 4.8.<br />
That’s why I couldn’t simply consume `WhisperNet` library in this project.<br />
Instead I implemented slightly different C# wrappers for the same C++ DLL.

Luckily, my [ComLightInterop](https://www.nuget.org/packages/ComLightInterop/) library is old,
and it still supports legacy .NET framework, along with other old versions of the runtime like .NET Core 2.1.