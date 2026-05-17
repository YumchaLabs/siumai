# Completion Metadata Boundary Convergence - Milestones

Last updated: 2026-05-17

## CMBC-M0 - Scope Locked

Acceptance criteria:

- The duplicated native/compatible completion metadata helpers are identified.
- The protocol seam target is documented.
- Non-goals prevent the lane from expanding into request or prompt behavior.

Status: completed

## CMBC-M1 - Shared Helper Exists

Acceptance criteria:

- The protocol crate exposes a reusable completion metadata helper.
- Unit tests prove logprobs, sources, namespacing, merge, and empty metadata behavior.

Status: completed

## CMBC-M2 - Provider Clients Use The Seam

Acceptance criteria:

- Native OpenAI completion no longer owns duplicate metadata helpers.
- OpenAI-compatible completion no longer owns duplicate metadata helpers.
- Provider-specific identity and usage parsing remain in provider clients.

Status: completed

## CMBC-M3 - Gates And Commit Complete

Acceptance criteria:

- Formatting and package test gates pass.
- Workstream evidence is updated.
- The refactor is committed with a conventional commit message.

Status: completed
