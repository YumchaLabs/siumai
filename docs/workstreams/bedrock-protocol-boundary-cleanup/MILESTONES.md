# Bedrock Protocol Boundary Cleanup - Milestones

Last updated: 2026-05-17

## M0 - Scope frozen

Acceptance criteria:

- Problem and non-goals are documented.
- First executable task is bounded.
- Evidence gates are named.

Status: completed

## M1 - Chat tests isolated

Acceptance criteria:

- Bedrock chat standard tests live in `chat/tests.rs`.
- `chat.rs` remains the production protocol shell plus `mod tests;`.
- Existing source guards still prove request/response metadata direction invariants.
- Focused Bedrock no-network tests pass.

Status: completed

## M2 - Production boundary decision

Acceptance criteria:

- Request planning / response shaping / stream conversion are either left in place with rationale or
  split into a smaller bounded task.
- No file-size-only split remains on the task ledger.

Status: completed

## M3 - Closeout

Acceptance criteria:

- Final gates are recorded.
- Remaining risks or follow-ons are explicit.
- Workstream status is closed or the next bounded task is identified.

Status: completed
