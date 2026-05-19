# Fearless Module Deepening — TODO

Status: Complete
Last updated: 2026-05-19

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## M0 — Scope And Evidence Freeze

- [x] FMD-010 [owner=planner] [deps=none] [scope=docs/workstreams/fearless-module-deepening]
  Goal: Freeze problem, target state, non-goals, evidence anchors, and task order for the module
  deepening lane.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree.
  Review: Planner self-review; no code changes in this task.
  Evidence: 2026-05-19 document presence and `WORKSTREAM.json` parse check recorded in
  `EVIDENCE_AND_GATES.md`.
  Handoff: DONE. FMD-020 is the first executable coding task.

## M1 — Spec Provider Residue Removal

- [x] FMD-020 [owner=codex] [deps=FMD-010] [scope=siumai-spec,siumai-core,siumai-protocol-openai,siumai-protocol-anthropic,siumai-protocol-gemini,siumai]
  Goal: Move provider-defined tool catalog helpers out of `siumai-spec` while keeping passive
  `Tool::ProviderDefined` data shapes stable.
  Validation:
  `cargo nextest run -p siumai-spec --no-default-features`;
  `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses`;
  `cargo nextest run -p siumai-protocol-anthropic --features anthropic-standard`;
  `cargo nextest run -p siumai-protocol-gemini --features google`;
  `cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast`.
  Review: `review-workstream` before accepting completion.
  Evidence: 2026-05-19 FMD-020 verification recorded in `EVIDENCE_AND_GATES.md`. Canonical
  catalogs now live in protocol/provider crates; `siumai-spec`/`siumai-core` source guards prevent
  catalog ownership from returning; facade import guards keep `siumai::tools::*` compatibility.
  Handoff: Keep old facade/provider_ext import paths working via re-exports where public surface
  compatibility requires it; canonical catalog ownership no longer lives in `siumai-spec`.

- [x] FMD-030 [owner=codex] [deps=FMD-020] [scope=siumai-spec,siumai-core,siumai-registry,siumai]
  Goal: Introduce a provider-id-first classification seam and demote closed `ProviderType` to
  compatibility or registry-owned policy.
  Validation:
  `cargo nextest run -p siumai-spec --no-default-features provider`;
  `cargo nextest run -p siumai-core --no-default-features core_provider_boundary_test`;
  `cargo nextest run -p siumai-registry --no-default-features`;
  `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast`.
  Review: 2026-05-19 `review-workstream` self-review completed; no blocking findings.
  Evidence: 2026-05-19 FMD-030 verification recorded in `EVIDENCE_AND_GATES.md`. `LlmClient`,
  `ClientWrapper`, core validation reports, and registry catalog lookups are provider-id-first;
  `ProviderType` is documented and isolated as legacy compatibility classification.
  Handoff: FMD-040 is next. Public `ProviderType` remains available for compatibility, but primary
  core/registry flows should not add new closed-enum dependencies.

## M2 — Compatibility Interface Isolation

- [x] FMD-040 [owner=codex] [deps=FMD-030] [scope=siumai-core,siumai-registry,siumai]
  Goal: Physically isolate `LlmClient` and generic-client downcast Interfaces under an explicit
  compatibility Module while keeping family-first registry handles unchanged.
  Validation:
  `cargo nextest run -p siumai-registry --no-default-features`;
  `cargo nextest run -p siumai-registry --features builtins --no-default-features`;
  `cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast`.
  Review: `review-workstream` before accepting completion.
  Evidence: 2026-05-19 FMD-040 verification recorded in `EVIDENCE_AND_GATES.md`. The physical
  `LlmClient`/`ClientWrapper` implementation now lives under `siumai_core::compat::client`; legacy
  core aliases remain migration-only, registry/facade compatibility paths are explicit, and guards
  prevent production imports from returning to broad `client` roots.
  Handoff: FMD-050 is next. Keep compatibility APIs under explicit `compat` naming and continue
  collapsing old generic `build_*_client` helper flows behind family-first registry construction.

- [x] FMD-050 [owner=codex] [deps=FMD-040] [scope=siumai-registry/src/registry/factory.rs,siumai-registry/src/provider,siumai-registry/src/registry/factories]
  Goal: Collapse or isolate old `build_*_client` helpers so new provider construction goes through
  provider-owned config plus `ProviderFactory` family methods.
  Validation:
  `cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast`.
  Review: `review-workstream` before accepting completion.
  Evidence: 2026-05-19 FMD-050 verification recorded in `EVIDENCE_AND_GATES.md`. OpenAI and
  Anthropic production factories no longer call legacy broad `registry::factory::build_*_client`
  helpers; those public helpers are compatibility-only deprecated shims; source guards prevent
  production factory call regressions.
  Handoff: FMD-060 is next. `Option<()>` placeholders remain only on deprecated compatibility
  helpers to avoid silently changing that public shim signature; new provider construction should
  stay inside provider-owned typed builders and family-first `ProviderFactory` methods.

## M3 — Protocol And Bridge Module Deepening

- [x] FMD-060 [owner=codex] [deps=FMD-020] [scope=siumai-bridge/src/request,siumai-bridge/src/contracts.rs,siumai-protocol-*]
  Goal: Split request normalization into narrower protocol-pair Adapters where behavior already
  varies, without changing bridge semantics.
  Validation:
  `cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast`;
  `cargo nextest run -p siumai --features openai,anthropic,google --test request_direct_bridge_fixtures_alignment_test --no-fail-fast`.
  Review: `review-workstream` before accepting completion.
  Evidence: 2026-05-19 FMD-060 verification recorded in `EVIDENCE_AND_GATES.md`. Gemini
  GenerateContent request JSON parsing moved behind
  `siumai-bridge/src/request/normalize/gemini_generate_content.rs`; `request/normalize.rs` now keeps
  only the public Gemini bridge wrappers and shared primitives for that slice. Source guards prevent
  the Gemini typed parser and tool/content policy from returning to the monolithic normalizer.
  Handoff: FMD-070 is next. Shared request primitives may remain common, but protocol-owned policy
  should keep moving behind narrow behavior seams; for FMD-070, focus on OpenAI protocol internals
  around request mapping/tools/output items/usage/stream accumulation rather than splitting by size.

- [x] FMD-070 [owner=codex] [deps=FMD-020] [scope=siumai-protocol-openai/src/standards/openai]
  Goal: Deepen OpenAI protocol internals around request mapping, hosted tools, output items,
  usage/metadata, and stream accumulation.
  Validation:
  `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`;
  targeted facade OpenAI fixture tests that cover chat completions, responses, tools, and streams.
  Review: 2026-05-19 `review-workstream` self-review completed; no blocking findings.
  Evidence: 2026-05-19 FMD-070 verification recorded in `EVIDENCE_AND_GATES.md`. OpenAI
  Responses request body assembly now lives behind `responses_request_builder`; response
  hosted/dynamic output helpers live behind `responses/hosted_tools.rs`; response provider
  metadata/source/logprobs aggregation lives behind `responses/metadata.rs`. Source guards prevent
  these policies from returning to the monolithic request/response transformer files. Protocol
  all-features and targeted facade OpenAI chat completions/responses/tools/streams fixture gates
  passed.
  Handoff: FMD-080 is next. Streaming accumulation already has narrower converter submodules and
  was covered by fixture gates; do not split it further in this lane unless a future behavior seam
  gives a better Interface than the existing converter modules.

## M4 — Legacy Content Directional Boundary

- [x] FMD-080 [owner=planner] [deps=FMD-060] [scope=siumai-spec,siumai-bridge,siumai-protocol-*,docs]
  Goal: Decide whether this lane should implement the first directional `ContentPart` adapter slice
  or open a narrower breaking-slice workstream.
  Validation: design note added to this workstream or a new workstream exists with explicit
  request/response/compat target state.
  Review: 2026-05-19 planner self-review; no code changes in this task.
  Evidence: 2026-05-19 decision note recorded in
  `FMD-080-content-part-directional-boundary-decision.md` and evidence recorded in
  `EVIDENCE_AND_GATES.md`. The lane will implement a non-breaking request-side bridge legacy
  `ContentPart` adapter extraction in FMD-090 instead of opening the breaking namespace-move child
  workstream now.
  Handoff: FMD-090 should extract request-side legacy `ContentPart` adapter helpers from
  `siumai-bridge/src/request/normalize.rs` into `siumai-bridge/src/request/legacy_content.rs`, add
  source guards, and run bridge request/facade fixture gates. Do not rename/remove the public legacy
  carrier in this lane.

- [x] FMD-090 [owner=codex] [deps=FMD-080] [scope=siumai-bridge]
  Goal: Implement the approved non-breaking request-side bridge legacy `ContentPart` adapter proof
  slice.
  Validation:
  `cargo fmt --check -p siumai-bridge`;
  `cargo nextest run -p siumai-bridge --features openai,anthropic,google request --no-fail-fast`;
  `cargo nextest run -p siumai --features openai,anthropic,google --test request_direct_bridge_fixtures_alignment_test --no-fail-fast`.
  `cargo nextest run -p siumai-spec --no-default-features prompt --no-fail-fast` was skipped
  because this non-breaking proof slice did not touch spec prompt projection helpers.
  Review: 2026-05-19 `review-workstream` self-review completed; no blocking findings.
  Evidence: 2026-05-19 FMD-090 verification recorded in `EVIDENCE_AND_GATES.md`. Request-side
  legacy `ContentPart` construction now lives in `siumai-bridge/src/request/legacy_content.rs`;
  OpenAI/Anthropic/Gemini request normalization route through that adapter; source guards prevent
  helper definitions and response `provider_metadata` population from drifting back into the
  monolithic request normalizer.
  Handoff: FMD-100 is next. Do not rename/remove public `ContentPart` in this lane; the breaking
  namespace move remains deferred until a dedicated compatibility-break workstream is opened.

## M5 — Test Locality And Closeout

- [x] FMD-100 [owner=codex] [deps=FMD-020,FMD-030] [scope=siumai/tests,siumai-registry/tests,docs]
  Goal: Refactor oversized architecture/parity tests into manifest or module-driven tests where it
  improves future provider-change locality.
  Validation:
  `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`;
  `cargo nextest run -p siumai --test provider_public_path_parity_test --no-fail-fast`;
  `cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast`.
  Review: 2026-05-19 `review-workstream` self-review completed; no blocking findings.
  Evidence: 2026-05-19 FMD-100 verification recorded in `EVIDENCE_AND_GATES.md`.
  `provider_public_path_parity_test.rs` was reduced from a 47k-line monolith to a shared harness
  plus provider-local modules under `siumai/tests/provider_public_path_parity/`; registry source
  guards now require the split and inspect provider modules through a manifest.
  Handoff: FMD-110 is next. Public-surface import tests remain broad by design; do not split them
  without a concrete failing-locality problem.

- [x] FMD-110 [owner=planner] [deps=FMD-050,FMD-060,FMD-070,FMD-080] [scope=docs/workstreams/fearless-module-deepening,docs/architecture,docs/adr]
  Goal: Close the lane or split remaining large/breaking work into narrower follow-ons.
  Validation: `verify-rust-workstream` records fresh final gate evidence.
  Review: 2026-05-19 closeout self-review found no blocking findings.
  Evidence: 2026-05-19 FMD-110 closeout verification recorded in `EVIDENCE_AND_GATES.md`;
  `WORKSTREAM.json` is marked complete and `HANDOFF.md` records residual follow-ons.
  Handoff: DONE. The lane is closed. Defer the breaking public `ContentPart` namespace move to a
  dedicated future compatibility-break workstream if/when ADR-0008 preconditions are met.
