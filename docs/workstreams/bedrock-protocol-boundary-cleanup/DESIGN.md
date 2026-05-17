# Bedrock Protocol Boundary Cleanup

Status: Active
Last updated: 2026-05-17

## Why This Lane Exists

The V4 core refactor is closed, and future cleanup should start from concrete follow-ons. Bedrock is
the highest-signal provider-local candidate because `standards/bedrock/chat.rs` is a large protocol
Module that owns request shaping, response conversion, JSON stream conversion, typed metadata guards,
and a large inline test suite.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
- Existing docs:
  - `docs/workstreams/fearless-refactor-v4/follow-ons.md`
  - `docs/workstreams/fearless-refactor-v4/provider-capability-alignment-matrix.md`
  - `docs/workstreams/fearless-refactor-v4/typed-metadata-boundary-matrix.md`
- Related workstreams:
  - `docs/workstreams/bedrock-embedding-alignment`
  - `docs/workstreams/bedrock-image-alignment`

## Problem

`siumai-provider-amazon-bedrock/src/standards/bedrock/chat.rs` is deep in behavior but too broad in
locality. Understanding the request transformer, response transformer, stream converter, and source
guards currently requires scanning one large file. That makes future Bedrock protocol changes harder
to review and increases the risk that tests or compatibility guards bury the production protocol
interface again.

## Target State

- The Bedrock chat standard keeps its public module path and behavior.
- Request shaping, response shaping, stream conversion, and contract tests each have clearer local
  owners.
- Test-only code is outside the production protocol shell.
- Source guards prevent request/response metadata direction regressions.
- No-network Bedrock package tests prove behavior did not change.

## In Scope

- Bedrock chat protocol standard modules under `siumai-provider-amazon-bedrock/src/standards/bedrock`.
- Test isolation and source guards for Bedrock chat protocol behavior.
- Mechanical module extraction only when it improves locality and preserves the public module path.

## Out Of Scope

- New Bedrock capabilities.
- Public API redesign.
- Changing Bedrock request/response wire shape.
- Reopening V4 core architecture.
- Cleaning Bedrock embedding/image/rerank unless a chat extraction reveals a shared helper with real
  leverage.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Inline Bedrock chat tests can move to a sibling test module without behavior changes. | High | Rust child test modules can access private parent items, and OpenAI-compatible used this pattern successfully. | Keep tests inline and choose a production-only extraction instead. |
| First valuable slice is test isolation, not request/response extraction. | Medium | The file's largest low-risk mixed concern is the inline test suite after `#[cfg(test)]`. | If tests need too much private access, split a smaller source-guard module first. |
| Bedrock chat behavior is already well covered by no-network tests. | High | Existing chat standard file carries extensive request, response, stream, and source-guard tests. | Add missing focused tests before moving production code. |

## Architecture Direction

Use small, behavior-preserving extractions that deepen Modules by increasing locality:

- `chat.rs` should remain the public standard shell.
- `chat/tests.rs` should own the Bedrock chat standard contract suite.
- Later slices may extract request planning, response metadata helpers, or stream conversion only if
  the new Module has a smaller Interface than the Implementation it hides.

Do not split by size alone. A new Module must improve locality or leverage.

## Closeout Condition

This lane can close when:

- Bedrock chat test ownership is no longer buried in the production protocol shell,
- any additional extracted production Modules have clear owners and source guards,
- focused Bedrock no-network gates pass,
- docs record any deferred follow-ons,
- and `WORKSTREAM.json` / `HANDOFF.md` describe the next step or closure.
