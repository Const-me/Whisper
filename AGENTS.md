# AGENTS — Guide for AI coding agents

Purpose
- Help AI agents quickly become productive in the Whisper repo by summarizing build steps, areas of responsibility, and development guardrails.

Quick start
- Build (Visual Studio 2022, Release): `msbuild WhisperCpp.sln /property:Configuration=Release`
- Before building shaders, run the C# shader compression tool: see [Tools/CompressShaders](Tools/CompressShaders)

Key areas
- `Whisper/` — native core and Direct3D integration
- `ComputeShaders/` — HLSL compute shaders and helpers
- `WhisperNet/` — managed wrapper and packaging
- `Examples/`, `Tools/` — samples and utilities

Agent behavior and conventions
- Platform: Windows-only. Target Visual Studio 2022 and Windows SDK.
- Prefer small, targeted changes. Preserve existing abstractions and naming conventions.
- When editing shaders, follow existing compute-shader structure and comments; prefer minimal, verifiable fixes.
- Do not add new dependencies without explicit approval.

Useful links
- Main solution: [WhisperCpp.sln](WhisperCpp.sln)
- Existing Copilot instructions: [.github/copilot-instructions.md](.github/copilot-instructions.md)

Suggested next customizations
- Create a small skill to automate shader compression and verify shader build outputs.
- Add an automated task to run `msbuild` and report top-level build errors.

---
This file is intentionally concise. For detailed developer guidance see [.github/copilot-instructions.md](.github/copilot-instructions.md) and the repository README files.
