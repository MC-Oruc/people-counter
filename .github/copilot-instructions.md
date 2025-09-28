---
description: people-counter — Copilot system prompt to verify code via compile + self-test after every change (cross-platform with .venv)
applyTo: '**/*'
---

# Copilot System Prompt — people-counter

Purpose: After any code change in this repository, always run a consistent verification pipeline. Use a Python virtual environment (`.venv`) and run commands from the repository root. Support Windows, Linux, and macOS.

## Core Rules

- Project root: repository root (`.`)
- Use relative paths only (no machine-specific absolute paths)
- Python virtual environment folder: `.venv`
- Run all commands from the repository root
- Support Windows (PowerShell/pwsh) and Linux/macOS (bash/zsh) variants

## Mandatory Verification Pipeline (after every code change)

1) Compile (fast syntax/symbol check)
   - Windows (PowerShell/pwsh):
     ```pwsh
     .\\.venv\\Scripts\\python -m compileall -q .
     ```
   - Linux/macOS (bash/zsh):
     ```bash
     ./.venv/bin/python -m compileall -q .
     ```
   - If this fails: summarize the error, fix it, and rerun. Do not end the task as “done” with a failing compile.

2) If compile passes, run self-test
   - Windows (PowerShell/pwsh):
     ```pwsh
     .\\.venv\\Scripts\\python -m people_counter self-test
     ```
   - Linux/macOS (bash/zsh):
     ```bash
     ./.venv/bin/python -m people_counter self-test
     ```
   - If exit code != 0 or there are clear error logs: diagnose, apply a small/low-risk fix, and rerun.
   - If successful, report briefly: `Compile: PASS, Self-test: PASS`.


## Reporting and Quality Gates

- At the end of each turn, provide a short “quality gates” summary:
  - Compile: PASS/FAIL
  - Self-test: PASS/FAIL
  - Brief summary of fixes performed
- Do not conclude the session as complete when compile/tests are failing. Fix and re-verify.

## Notes

- In PowerShell and shells, keep each command on a single line.
- These rules are only for the verification routine; normal application runs can be performed as requested by the user.
