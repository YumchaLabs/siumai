# Fearless Residual Architecture Deepening - TODO

Status: Active
Last updated: 2026-05-27

## M0 - Planning

- [x] FRAD-010 [owner=planner] [deps=none] [scope=docs/workstreams/fearless-residual-architecture-deepening,docs/workstreams/INDEX.md]
  Goal: Open the durable workstream for all five residual architecture deepening candidates.
  Validation: `python -m json.tool docs/workstreams/fearless-residual-architecture-deepening/WORKSTREAM.json`; `git diff --check -- docs/workstreams/fearless-residual-architecture-deepening docs/workstreams/INDEX.md`.
  Review: Confirm the lane references ADR-0001, ADR-0002, ADR-0006, ADR-0007, ADR-0008, and ADR-0009 without contradicting them.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`.
  Handoff: DONE. First executable code task is FRAD-020.

## M1 - Registry Provider Descriptor Seam

- [x] FRAD-020 [owner=codex] [deps=FRAD-010] [scope=siumai-registry/src/provider_catalog.rs,siumai-registry/src/native_provider_metadata.rs,siumai-registry/src/provider/catalog_ids.rs,siumai-registry/src/registry/helpers.rs,siumai-registry/src/registry/mod.rs,siumai-registry/src/registry/factories,CHANGELOG.md,siumai-registry/CHANGELOG.md]
  Goal: Collapse provider identity, aliases, metadata, model lists, default-model policy, feature-gated factory resolution, and catalog views behind one registry-owned descriptor seam.
  Validation: `cargo fmt --check -p siumai-registry`; `cargo nextest run -p siumai-registry provider_catalog --no-fail-fast`; `cargo nextest run -p siumai-registry factory_architecture_boundary_test --no-fail-fast`.
  Review: Confirm adding one built-in provider no longer requires editing unrelated catalog, helper, and test matrices with repeated provider facts.
  Evidence: `EVIDENCE_AND_GATES.md`, registry changelog.
  Handoff: DONE. `registry::provider_descriptor` owns built-in provider default-model lookup,
  factory resolution, and enabled-factory registration; `ProviderCatalogDescriptor` owns catalog
  projection into public `ProviderInfo`, and source guards prevent `helpers.rs` from growing
  concrete provider factory facts again.

## M2 - Protocol And Bridge Conversion Seams

- [x] FRAD-030 [owner=codex] [deps=FRAD-010] [scope=siumai-protocol-openai/src/standards/openai,CHANGELOG.md,siumai-protocol-openai/CHANGELOG.md]
  Goal: Deepen OpenAI-compatible message dialect conversion so `utils.rs` no longer owns tools, messages, dialect-specific conversion, response formats, finish reasons, and usage as one broad Interface.
  Validation: `cargo fmt --check -p siumai-protocol-openai`; `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses openai --no-fail-fast`.
  Review: Confirm Perplexity, DeepSeek, xAI, Mistral, OpenAI Chat, and Responses conversion still preserve provider-specific behavior through dialect-focused tests.
  Evidence: `EVIDENCE_AND_GATES.md`, protocol changelog.
  Handoff: DONE. `utils::message_dialect` now owns OpenAI-compatible, OpenAI Chat,
  Perplexity, DeepSeek, xAI, and Mistral message conversion plus dialect-local tests; `utils::*`
  remains a compatibility re-export surface for existing call sites.

- [x] FRAD-040 [owner=codex] [deps=FRAD-030] [scope=siumai-bridge/src/request,siumai-bridge/src/request/tests.rs,CHANGELOG.md,siumai-bridge/CHANGELOG.md]
  Goal: Split bridge request normalization into per-wire-format codec Modules while keeping public bridge helper functions stable.
  Validation: `cargo fmt --check -p siumai-bridge`; `cargo nextest run -p siumai-bridge --features openai,anthropic,google request --no-fail-fast`.
  Review: Confirm OpenAI Responses, OpenAI Chat Completions, Anthropic Messages, and Gemini GenerateContent fixtures still normalize to the same `ChatRequest` shape.
  Evidence: `EVIDENCE_AND_GATES.md`, bridge changelog.
  Handoff: DONE. Request normalization now delegates to `normalize::openai_responses`,
  `normalize::openai_chat_completions`, `normalize::anthropic_messages`, and the existing
  `normalize::gemini_generate_content` codec shim while preserving the public bridge helpers.

## M3 - Test Harness Depth

- [x] FRAD-050 [owner=codex] [deps=FRAD-020] [scope=siumai-registry/src/registry/factories/contract_tests.rs,siumai-registry/tests/factory_architecture_boundary_test.rs,siumai/tests/provider_public_path_parity,siumai/tests/public_surface_imports_test.rs,CHANGELOG.md,crate changelogs]
  Goal: Replace manual provider contract/public-path matrices with scenario/harness seams where the test Interface is smaller than the implementation.
  Validation: `cargo fmt --check -p siumai-registry -p siumai`; `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-fail-fast`; targeted public path parity tests for touched providers.
  Review: Confirm provider behavior coverage is not reduced and future providers can add scenarios without copying large test blocks.
  Evidence: `EVIDENCE_AND_GATES.md`, changelogs.
  Handoff: DONE. Factory family override requirements now use named
  `FactoryFamilyOverrideContract` scenarios, provider public-path architecture guards use a
  `ProviderPublicPathModule` manifest object, and root public-path parity helpers centralize
  built-in provider map/builder setup through `BuiltInProviderRegistryHarness`.

## M4 - ADR-0008 ContentPart Decision Or Move

- [x] FRAD-060 [owner=codex] [deps=FRAD-010,FRAD-040] [scope=siumai-spec/src/types,siumai-core/src,siumai/src,siumai-protocol-*/src,CHANGELOG.md,crate changelogs]
  Goal: Prove ADR-0008 root-move gates and complete the remaining low-level `ContentPart` root compatibility move, or record the exact blocking gate with executable evidence.
  Validation: `cargo fmt --check -p siumai-spec -p siumai-core -p siumai`; `cargo nextest run -p siumai-spec content --no-fail-fast`; `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`; provider/protocol fixture parity gates identified during implementation.
  Review: Confirm request-side code uses prompt parts, response-side code uses generated output parts, and compatibility payload serde remains stable.
  Evidence: `EVIDENCE_AND_GATES.md`, ADR-0008 gate record, changelogs.
  Handoff: DONE_WITH_CONCERNS. High-value production legacy `ContentPart` usage now imports through
  explicit compatibility namespaces, and ADR/workstream evidence records that the low-level root
  move remains blocked by serde-facing `ChatMessage` / `ChatResponse` identity, blanket
  spec/core root re-exports, provider/protocol/bridge response parity, and a missing full-root-move
  fixture suite.

## M5 - Review And Closeout

- [ ] FRAD-070 [owner=planner] [deps=FRAD-020,FRAD-030,FRAD-040,FRAD-050,FRAD-060] [scope=docs/workstreams/fearless-residual-architecture-deepening,CHANGELOG.md,crate changelogs]
  Goal: Review, verify, close, or split only concrete residual follow-ons.
  Validation: `review-workstream` has no blocking findings; `verify-rust-workstream` records fresh final gates.
  Review: Confirm workstream docs, evidence, and changelogs agree with the final architecture.
  Evidence: `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`.
  Handoff: TBD.
