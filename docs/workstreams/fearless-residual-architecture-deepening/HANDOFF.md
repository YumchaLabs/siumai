# Fearless Residual Architecture Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open for five residual architecture-review candidates. FRAD-010 through FRAD-060
are complete. The next executable task is FRAD-070, review and closeout.

## Active Task

- Task ID: FRAD-070
- Owner: codex
- Files:
  - `docs/workstreams/fearless-residual-architecture-deepening`
  - `CHANGELOG.md`
  - crate changelogs
- Validation:
  - `review-workstream`
  - `verify-rust-workstream` final gates recorded in `EVIDENCE_AND_GATES.md`
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
- FRAD-040 split bridge request normalization into per-wire-format codec modules for OpenAI
  Responses, OpenAI Chat Completions, Anthropic Messages, and Gemini GenerateContent. The parent
  `normalize.rs` now keeps public wrapper functions, request hook/loss-policy flow, and shared
  helpers.
- FRAD-050 deepened test harnesses: factory family override requirements are named
  `FactoryFamilyOverrideContract` scenarios, provider public-path source guards use a
  `ProviderPublicPathModule` manifest object, and built-in registry parity setup crosses
  `BuiltInProviderRegistryHarness`.
- FRAD-060 moved high-value production legacy `ContentPart` usage to explicit compat imports and
  recorded `FRAD-060-content-part-root-move-decision.md`: the low-level root move remains blocked
  until a full provider/protocol/bridge root-move fixture suite exists.

## Next Recommended Action

Commit FRAD-060, then run FRAD-070 review and closeout.
