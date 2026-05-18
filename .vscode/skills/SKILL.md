title: Create SKILL.md (agent-customization)
scope: workspace
tags: [onboarding, automation, agent-customization]
created: 2026-05-19
---

# SKILL: Create SKILL.md (agent-customization)

Purpose
- Provide a repeatable, workspace-scoped procedure to convert a multi-step conversational workflow into a `SKILL.md` that agents can run.

Scope
- Workspace-scoped: saved to `.vscode/skills/SKILL.md` by default. Can be adapted for personal scope.

When to run
- After a chat produces a reproducible multi-step process (debugging, onboarding, release steps).

Inputs
- Conversation transcript or summary.
- Desired artifact and location (patch, file, test, PR template).

Outputs
- A completed `SKILL.md` in `.vscode/skills/`.
- A short example prompt and a minimal conversation snippet to exercise the skill.

Step-by-step process
1. Extract
   - Read the conversation and identify ordered steps, decision points, tools, and file paths.
2. Clarify
   - Ask any missing questions needed to make steps deterministic (see Clarifying Questions).
3. Draft
   - Produce a concise SKILL.md with purpose, scope, inputs/outputs, and explicit steps.
4. Iterate
   - Save the draft, run one example execution (manual or automated), note gaps, and update.
5. Publish
   - Commit the SKILL and add a short usage example to the repo README or developer docs.

Clarifying Questions (examples)
- Should this SKILL be workspace-scoped (`.vscode/skills`) or personal?
- What is the exact expected artifact path and filename?
- Are there environment prerequisites (OS, SDKs, tool versions, credentials)?

Quality checks
- Steps are concrete and ordered.
- Decision points include explicit branching (yes/no) and next actions.
- Completion criteria are verifiable (file created, test passed, PR opened).

Example conversation snippet
```
User: We have a flaky transcription test that fails intermittently.
Agent: Extract the test steps, reproduce locally, and add a deterministic fixture.
User: Produce a SKILL.md that formalizes these steps and includes the fixture path.
```

Example prompts
- "Create a SKILL.md to formalize our transcription test reproduction steps."
- "Turn this debugging chat into a workspace SKILL that produces a test fixture and CI patch."

How to run this SKILL (manual)
1. Collect the conversation or notes that describe the workflow.
2. Run the steps in the SKILL to draft the file.
3. Validate by walking through the example prompt and ensuring the outputs are produced.

Suggested follow-ups
- Add a short test harness or example input files to exercise the SKILL automatically.
- Add `metadata` and `tags` for discoverability by other agents.

Notes
- Keep SKILLs focused and deterministic. Link to extended docs for complex workflows.

----

Saved: `.vscode/skills/SKILL.md`
