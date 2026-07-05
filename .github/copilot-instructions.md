# Copilot instructions for the Whisper repository

## Project context
- This repository contains a Windows-only speech transcription engine built around DirectCompute / HLSL, C++, C#, and a small amount of PowerShell.
- The main native project is the Visual Studio solution at WhisperCpp.sln.
- The repository is focused on Windows desktop scenarios, so keep changes compatible with Visual Studio 2022 and the Windows SDK.

## Preferred working style
- Prefer small, targeted changes over broad rewrites.
- Reuse existing abstractions and naming patterns already used in the surrounding code.
- Preserve platform-specific behavior and avoid introducing cross-platform assumptions unless the user explicitly asks for them.
- When modifying shader code, keep the change consistent with the existing compute-shader structure and comments.

## Build and verification guidance
- The expected build flow is Visual Studio 2022 with the Release configuration.
- The C# tool project under Tools/CompressShaders is often required before building the native shaders pipeline.
- When changing native code, prefer checking the relevant project files and the surrounding headers/source before editing.
- If a full build cannot be run in the current environment, say so clearly and describe the most relevant verification you were able to perform.

## Repository areas
- Whisper/: core native implementation and Direct3D integration.
- ComputeShaders/: HLSL shaders and shader-related helpers.
- WhisperNet/: managed wrapper and packaging concerns.
- Examples/: sample applications and integration examples.
- Tools/: utilities such as shader compression and trace comparison tools.

## Guardrails
- Do not remove or weaken existing error handling unless the change specifically targets it.
- Do not introduce unnecessary dependencies.
- Keep comments concise and useful.
- For user-facing changes, preserve the project’s existing terminology and documentation style.
