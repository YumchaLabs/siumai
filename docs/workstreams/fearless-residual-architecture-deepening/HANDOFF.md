# Fearless Residual Architecture Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open for five residual architecture-review candidates. FRAD-010, FRAD-020, and
FRAD-030 are complete. The next executable task is FRAD-040, the bridge request codec split.

## Active Task

- Task ID: FRAD-040
- Owner: codex
- Files:
  - `siumai-bridge/src/request`
  - `siumai-bridge/src/request/tests.rs`
  - `CHANGELOG.md`
  - `siumai-bridge/CHANGELOG.md`
- Validation:
  - `cargo fmt --check -p siumai-bridge`
  - `cargo nextest run -p siumai-bridge --features openai,anthropic,google request --no-fail-fast`
- Status: READY
- Review: not started
- Evidence: `EVIDENCE_AND_GATES.md`

## Decisions

- Use one durable workstream for all five candidates because they share the same post-AI-SDK
  residual architecture review and closeout gate.
- Do not open a new ADR yet. Existing ADRs cover the direction; `ContentPart` remains controlled by
  ADR-0008.
- Execute in dependency order: registry descriptor first, protocol dialects second, bridge codecs
  third, test harness fourth, `ContentPart` gate/move fifth.
- FRAD-020 concentrated built-in provider default-model lookup, factory resolution, and enabled
  factory registration in `registry::provider_descriptor`; the public helper functions now cross
  that seam, and catalog projection uses `ProviderCatalogDescriptor`.
- FRAD-030 moved OpenAI-compatible, OpenAI Chat, Perplexity, DeepSeek, xAI, and Mistral message
  conversion into `utils::message_dialect` with dialect-local tests, while `utils::*` keeps stable
  compatibility re-exports for existing call sites.

## Next Recommended Action

Commit FRAD-030, then run FRAD-040 with the bridge request codec split as the bounded scope.
