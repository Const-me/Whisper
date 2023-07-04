The `WhisperPS.csproj` project in this folder builds PowerShell wrapper for Whisper.

I wouldn’t call the wrapper particularly great, but it works on my computer.<br/>
This should handle use cases like “transcribe all files in a directory” or “export multiple formats”.

The supported PowerShell version is 5.1, the one I have preinstalled on my Windows 10 computer.<br/>
I wouldn’t expect it to work with the newer PowerShell Core, the runtime is different.

## Installation

To install from [PowerShell Gallery](https://www.powershellgallery.com/) for the current user,
open Windows PowerShell, and run the following command:<br/>
`Install-Module -Name WhisperPS -Scope CurrentUser`

To install from PowerShell Gallery for all users,
open Windows PowerShell **as an administrator**, and run the following command:<br/>
`Install-Module -Name WhisperPS`

To install manually from github.com,
download WhisperPS.zip from [Releases](https://github.com/Const-me/Whisper/releases) page of this repository,
and extract into the following folder:<br/>
`%USERPROFILE%\Documents\WindowsPowerShell\Modules`<br/>
Create that folder if you don’t yet have it.

## Models

The binary files with the models are available for free download on [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main).<br/>
I recommend `ggml-medium.bin` (1.42GB in size, but that web page says 1.53 GB), because I’ve mostly tested the software with that model.<br/>
Compressed models in ZIP format with `mlmodelc` in the file name are not supported.

## Usage Example

```
Import-Module WhisperPS -DisableNameChecking
$Model = Import-WhisperModel D:\Data\Whisper\ggml-medium.bin
cd C:\Temp\2remove\Whisper
$Results = dir .\* -include *.wma, *.wav | Transcribe-File $Model
foreach ( $i in $Results ) { $txt = $i.SourceName + ".txt"; $i | Export-Text $txt; }
foreach ( $i in $Results ) { $txt = $i.SourceName + ".ts.txt"; $i | Export-Text $txt -timestamps; }
```

## Commands

Here’s the list of commands implemented by this module.

* `Get-Adapters` prints names of the graphics adapters visible to DirectCompute.<br/>
You can use these strings for the `-adapter` argument of the `Import-WhisperModel` command.
* `Import-WhisperModel` loads the model from disk, returns the object which keeps the model
* `Transcribe-File` loads audio file from disk, and transcribes the audio into text.
It returns the object which keeps both source file name, and transcribed text.
* `Export-Text` saves transcribed text into a text file, with or without timestamps.
* `Export-SubRip` saves transcribed text in *.srt format.
* `Export-WebVTT` saves transcribed text in *.vtt format.

You can use `man` (an alias for `get-help`) for detailed documentation on specific commands, example:<br/>
`man Import-WhisperModel -full`

## Miscellaneous

By default, PowerShell doesn’t print any informational or debug messages.<br/>
If you want these messages, run these commands:

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