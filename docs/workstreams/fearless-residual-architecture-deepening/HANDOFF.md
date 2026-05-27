# Fearless Residual Architecture Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open for five residual architecture-review candidates. FRAD-010 through FRAD-050
are complete. The next executable task is FRAD-060, the ADR-0008 `ContentPart` root compatibility
decision or move.

## Active Task

- Task ID: FRAD-060
- Owner: codex
- Files:
  - `siumai-spec/src/types`
  - `siumai-core/src`
  - `siumai/src`
  - `siumai-protocol-*/src`
  - `siumai/tests/public_surface_imports_test.rs`
  - `CHANGELOG.md`
  - crate changelogs
- Validation:
  - `cargo fmt --check -p siumai-spec -p siumai-core -p siumai`
  - `cargo nextest run -p siumai-spec content --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`
  - provider/protocol fixture parity gates identified during implementation
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

## Next Recommended Action

Commit FRAD-050, then run FRAD-060 by proving ADR-0008 root-move gates before changing any
`ContentPart` compatibility path.
