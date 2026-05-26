# Fearless AI SDK Contract Hardening - Milestones

Status: Active
Last updated: 2026-05-26

## M0 - Scope And Evidence Freeze

Exit criteria:

- The 12 contract gaps are recorded.
- AI SDK reference files and Siumai authority docs are linked.
- Changelog tracking is required by the task ledger.
- `WORKSTREAM.json` points to the first executable task.

Primary evidence:

- `DESIGN.md`
- `TODO.md`
- `WORKSTREAM.json`
- `CHANGELOG.md`

## M1 - Stream Replay And Cancellation Contracts

Exit criteria:

- `StreamEnd.response.content` replay/fallback semantics are documented and guarded.
- A reasoning/text separation fixture proves terminal-only final text is preserved or the missing
  provider proof is split into a follow-on.
- `stream_with_cancel` is documented as the cancelable entry, with local and remote cancellation
  semantics separated.

Primary gates:

- `cargo nextest run -p siumai-spec stream --no-fail-fast`
- `cargo nextest run -p siumai-core streaming::processor --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai responses_sse --no-fail-fast`

## M2 - Diagnostics, Tools, Usage, And Capability Contracts

Exit criteria:

- Raw/private diagnostics ownership is explicit for provider metadata, raw/custom events,
  `ResponseMetadata`, and error detail.
- Public error messages are safe by contract or renamed to make safety explicit.
- Tool name validation, provider-executed ownership, and `ToolInputStart` stable fields are guarded.
- Stream usage snapshot semantics are documented and tested.
- Unsupported capability behavior has a shared contract and provider-specific exceptions are split.

Primary gates:

- `cargo nextest run -p siumai-spec tools --no-fail-fast`
- `cargo nextest run -p siumai-spec usage --no-fail-fast`
- `cargo nextest run -p siumai-core error --no-fail-fast`
- Targeted provider/protocol gates for touched fixtures.

## M3 - Integration And Closeout

Exit criteria:

- `review-workstream` finds no blocking workstream or code-quality issue.
- `verify-rust-workstream` records fresh focused gates and an appropriate broader gate.
- Root and touched crate changelogs describe the shipped contract changes.
- `WORKSTREAM.json`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, and `HANDOFF.md` agree.

Primary gates:

- `cargo fmt --check`
- Focused package `cargo nextest` gates for touched crates.
- `git diff --check -- CHANGELOG.md docs/workstreams/fearless-ai-sdk-contract-hardening docs/workstreams/INDEX.md`
