# Compress Shaders — Agent Skill

Purpose
- Automate building and running the C# shader compression tool under `Tools/CompressShaders`, and verify expected compressed shader outputs are produced for the `ComputeShaders/` pipeline.

When to use
- Before building or testing native code that depends on preprocessed/packed shaders.

Commands
- Build & run the tool (Release):

```bash
dotnet run --project Tools/CompressShaders/CompressShaders.csproj --configuration Release --no-launch-profile --verbosity minimal
```

- Verify success: expect the tool to write or update compressed shader artifacts (see `ComputeShaders/Readme.txt` and `Tools/CompressShaders/Readme.txt`). If the tool prints errors, surface them and stop.

Agent behavior
- Run the tool in a shell task. If it returns non-zero, collect stdout/stderr and fail the automation step.
- If compression succeeds, run a lightweight validation: ensure at least one compressed shader file timestamp changed under `ComputeShaders/` or output folder specified by the tool's README.
- Do not modify source shaders automatically. If changes are required, create a concise PR with the minimal edits and a test run summary.

Notes for humans
- The tool is a small C# console project: `Tools/CompressShaders/CompressShaders.csproj`.
- Running locally may require the appropriate .NET SDK; prefer `dotnet run` which will fail fast if the SDK is missing.
