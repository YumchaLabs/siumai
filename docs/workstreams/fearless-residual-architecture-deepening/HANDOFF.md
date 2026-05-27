# Fearless Residual Architecture Deepening - Handoff

Status: Active
Last updated: 2026-05-27

## Current State

The workstream is open for five residual architecture-review candidates. FRAD-010 through FRAD-040
are complete. The next executable task is FRAD-050, the provider contract/public-path harness
deepening.

## Active Task

- Task ID: FRAD-050
- Owner: codex
- Files:
  - `siumai-registry/src/registry/factories/contract_tests.rs`
  - `siumai-registry/tests/factory_architecture_boundary_test.rs`
  - `siumai/tests/provider_public_path_parity`
  - `siumai/tests/public_surface_imports_test.rs`
  - `CHANGELOG.md`
  - crate changelogs
- Validation:
  - `cargo fmt --check -p siumai-registry -p siumai`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-fail-fast`
  - targeted public path parity tests for touched providers
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

## Next Recommended Action

Commit FRAD-040, then run FRAD-050 with provider contract and public path harnesses as the bounded
scope.
