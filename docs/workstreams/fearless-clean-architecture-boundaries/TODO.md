# Fearless Clean Architecture Boundaries — TODO

Status: Closed
Last updated: 2026-05-21

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[!]` blocked or needs maintainer decision

## M0 — Scope And Evidence Freeze

- [x] FCAB-010 [owner=planner] [deps=none] [scope=docs/workstreams/fearless-clean-architecture-boundaries]
  Goal: Open the workstream, freeze the problem statement, target seams, non-goals, execution order,
  and evidence anchors.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree.
  Review: planner self-review; no code changes.
  Evidence: `docs/workstreams/fearless-clean-architecture-boundaries/DESIGN.md`.
  Handoff: First executable task is FCAB-020.

- [x] FCAB-020 [owner=codex] [deps=FCAB-010] [scope=docs,tests/source-guards,siumai-core,siumai-registry,siumai-bridge,siumai-protocol-*,siumai-provider-*,siumai]
  Goal: Produce a current seam inventory and add/refresh source guards that prevent known
  regressions before moving code.
  Validation:
  - `cargo nextest run -p siumai-registry provider_factory_generic_client_paths_are_explicit_compatibility_aliases stable_registry_handles_do_not_use_compat_client_paths_for_primary_family_execution --no-fail-fast`
  - `cargo nextest run -p siumai --test facade_architecture_boundary_test fearless_clean_architecture_inventory_tracks_current_guard_surfaces --no-fail-fast`
  - `cargo fmt --package siumai-registry --package siumai -- --check`
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-020-review.md`.
  Evidence: `docs/workstreams/fearless-clean-architecture-boundaries/seam-inventory.md` and
  `EVIDENCE_AND_GATES.md`.
  Handoff: Parallel lanes are identified in `seam-inventory.md`.

## M1 — Registry And Compatibility Construction Isolation

- [x] FCAB-030 [owner=codex] [deps=FCAB-020] [scope=siumai-registry/src/registry/entry/factory.rs,siumai-registry/src/registry/entry/handles,siumai-registry/src/provider,siumai-core/src/compat]
  Goal: Split the stable family construction Interface from compatibility `LlmClient`
  construction. Keep custom provider integration usable, but make compatibility paths explicit and
  non-primary.
  Validation:
  - `cargo nextest run -p siumai-registry --no-fail-fast registry::entry`
  - `cargo nextest run -p siumai-registry provider_factory_facets_split_stable_compat_and_extension_execution provider_factory_generic_client_paths_are_explicit_compatibility_aliases stable_registry_handles_do_not_use_compat_client_paths_for_primary_family_execution remaining_registry_handle_compat_paths_are_extension_only --no-fail-fast`
  - `cargo check -p siumai-registry --tests`
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-030-review.md`.
  Evidence: registry facets, source guards, architecture docs, and `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-040 should now convert built-in provider factory internals toward the narrowed
  facet story and delete redundant glue.

- [x] FCAB-040 [owner=codex] [deps=FCAB-030] [scope=siumai-registry/src/registry/factories,siumai-provider-*/src/providers,docs/migration,docs/architecture]
  Goal: Convert built-in provider factories to the narrowed family-first construction story and
  remove redundant compatibility glue where native family objects exist.
  Validation:
  - `cargo check -p siumai-registry --features openai,azure,anthropic,google,google-vertex,groq,xai,deepseek,deepinfra,minimaxi,ollama,cohere,togetherai,bedrock --lib`
  - `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test promoted_openai_compatible_vendor_factories_centralize_projection_glue builtin_provider_factories_centralize_typed_client_arc_projection openai_factory_centralizes_family_projection_glue openai_compatible_factory_centralizes_checked_family_projection_glue hybrid_provider_composite_clients_are_compat_only_adapters --no-fail-fast`
  - `cargo nextest run -p siumai-registry --features openai,azure,anthropic,google,google-vertex,groq,xai,deepseek,deepinfra,minimaxi,ollama,cohere,togetherai,bedrock togetherai_factory_supports_native_completion_family_path deepinfra_factory_materializes_unified_language_client fireworks_provider_factory_declares_unified_capabilities fireworks_registry_image_handle_prefers_provider_specific_build_overrides azure_factory_supports_native_text_family_path vertex_maas_factory_supports_completion_and_embedding_family_paths anthropic_vertex_factory_supports_native_text_family_path vertex_factory_supports_native_text_family_path deepseek_factory_returns_provider_owned_client ollama_factory_returns_provider_owned_client --no-fail-fast`
  - `cargo fmt --package siumai-registry --package siumai -- --check`
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-040-review.md`.
  Evidence: built-in provider factory projection helpers, source guards, architecture docs, and
  `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-050/060 can now deepen OpenAI-compatible runtime/vendor ownership without first
  untangling repeated registry projection glue.

## M2 — OpenAI-Compatible Protocol/Runtime/Vendor Deepening

- [x] FCAB-050 [owner=codex] [deps=FCAB-020] [scope=siumai-protocol-openai,siumai-provider-openai-compatible,siumai-provider-openai]
  Goal: Define and implement a single OpenAI-compatible runtime seam: protocol conversion stays
  protocol-owned, runtime adapter behavior is centralized, and provider crates stop mirroring the
  same responsibilities.
  Validation:
  - `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast`
  - `cargo nextest run -p siumai-provider-openai --features openai openai_completion_non_stream_uses_completions_body_shape openai_completion_stream_uses_completions_sse_shape openai_completion_stream_preserves_empty_and_whitespace_text_deltas openai_completion_stream_raw_chunks_follow_stream_start_before_response_metadata openai_completion_parse_error_emits_stream_start_before_error_without_raw_chunks --no-fail-fast`
  - `cargo fmt --package siumai-protocol-openai --package siumai-provider-openai-compatible --package siumai-provider-openai -- --check`
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-050-review.md`.
  Evidence: protocol-owned completion conversion module, runtime delegation source guard
  `openai_compatible_completion_streaming_conversion_is_protocol_owned`, fixture tests, and
  `EVIDENCE_AND_GATES.md`.
  Handoff: Compatibility re-export paths remain documented shims; FCAB-060 owns vendor cleanup.

- [x] FCAB-060 [owner=codex] [deps=FCAB-050] [scope=siumai-registry/src/registry/factories,siumai-provider-{groq,xai,deepseek,togetherai},siumai/src/provider_ext]
  Goal: Rewire OpenAI-compatible vendors so vendor Modules own only presets, quirks, typed options,
  metadata, and facade extension exports.
  Validation:
  - focused `cargo nextest run -p siumai-registry --features groq,xai,deepseek,togetherai --no-fail-fast`
  - public surface compile guards for touched providers.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-060-review.md`.
  Evidence: TogetherAI provider-owned image runtime, registry source guard
  `togetherai_provider_crate_owns_image_runtime`, vendor factory/public-surface tests, and
  `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-070 can now deepen directional content exports; no FCAB-060 long-tail vendor split
  is required.

## M3 — Directional Content Seam Deepening

- [x] FCAB-070 [owner=codex] [deps=FCAB-020] [scope=siumai-spec/src/types,siumai-core/src/types.rs,siumai/src/compat.rs,siumai/src/lib.rs]
  Goal: Make request prompt parts, generated output parts, and legacy compatibility content visibly
  separate in spec/core/facade exports. Reduce broad root exports that teach new code to use
  compatibility carriers.
  Validation:
  - `cargo nextest run -p siumai-spec --no-fail-fast content`
  - facade public-surface import guards.
  Review: `review-workstream`; verify compatibility imports remain documented.
  Evidence: directional `content::{prompt, output, compat}` namespaces, facade source guards,
  public-surface compile guards, and `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-080 should migrate production direct `ContentPart` construction behind named
  request/response adapters, starting from `siumai-bridge/src/response`,
  `siumai-protocol-{anthropic,gemini}`, and provider response/request adapters.

- [x] FCAB-080 [owner=codex] [deps=FCAB-070] [scope=siumai-protocol-*,siumai-provider-*,siumai-bridge/src/response,siumai-bridge/src/stream]
  Goal: Move remaining production direct `ContentPart` construction behind named request or response
  adapters, or replace it with directional prompt/output parts where lossless.
  Validation:
  - protocol-focused nextest filters for OpenAI, Anthropic, Gemini, Bedrock/Gemini provider paths.
  - source guards proving request adapters do not read response metadata and response adapters do
    not emit request provider options.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-080-review.md`.
  Evidence: `content-part-adapter-audit.md`, adapter source guards, focused provider/protocol tests,
  and `EVIDENCE_AND_GATES.md`.
  Handoff: Remaining direct `ContentPart` appearances are classified in
  `content-part-adapter-audit.md`; FCAB-090 can start provider-utils isolation without reopening the
  directional content adapter slice.

## M4 — Core Provider-Utils Isolation

- [x] FCAB-090 [owner=codex] [deps=FCAB-020,FCAB-030] [scope=siumai-core/src/{execution,streaming,utils,retry,encoding},Cargo.toml,docs/architecture]
  Goal: Decide and implement the first provider-utils isolation step: either a new
  `siumai-provider-utils` crate or a deep internal provider-utils module with a documented crate
  split path.
  Validation:
  - `cargo nextest run -p siumai-core --no-fail-fast`
  - compile checks for provider/protocol crates that import moved utilities.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-090-review.md`;
  verify the new crate owns implementation while old core utility modules are migration aliases.
  Evidence: `siumai-provider-utils`, `provider_utils_crate_owns_high_churn_provider_helpers`,
  provider/protocol compile gates, and `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-100 should continue moving remaining provider/protocol imports to
  `siumai-provider-utils` and classify leftover `siumai-core::utils` exports as stable,
  experimental, or compat before deleting aliases.

- [x] FCAB-100 [owner=codex] [deps=FCAB-090] [scope=siumai-core,siumai-protocol-*,siumai-provider-*]
  Goal: Move protocol/provider utility imports to the provider-utils seam and delete obsolete
  re-exports from `siumai-core` where migration docs permit.
  Validation:
  - focused package gates for touched provider/protocol crates.
  - `cargo nextest run -p siumai-core --no-fail-fast`
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-100-review.md`.
  Evidence: `provider-utils-classification.md`, expanded provider-utils crate tests, import source
  guards, and `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-110 can move to bridge target adapters. Remaining core-owned `utils` modules are
  classified as stable core (`cancel`) or explicit compat (`streaming_tool_call`); matching moved
  utility modules are compatibility aliases backed by `siumai-provider-utils`.

## M5 — Bridge Contract Deepening

- [x] FCAB-110 [owner=codex] [deps=FCAB-020,FCAB-080] [scope=siumai-bridge,siumai-protocol-openai,siumai-protocol-anthropic,siumai-protocol-gemini]
  Goal: Move target-specific request/response/stream adapters out of `siumai-bridge` into
  protocol-owned modules where feasible. Keep bridge reports, policies, lifecycle, and customization
  in `siumai-bridge`.
  Validation:
  - `cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast`
  - protocol package gates for moved adapters.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-110-review.md`;
  verify `siumai-bridge` no longer owns the moved Gemini GenerateContent request parser and retained
  bridge-owned adapters are documented.
  Evidence: `bridge-target-adapter-audit.md`, refreshed Gemini source guards, and
  `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-120 can proceed to facade/public-surface tightening. Remaining bridge-owned
  OpenAI/Anthropic direct-pair and normalization code is listed in `bridge-target-adapter-audit.md`
  and should move only when bridge loss/replay policy can stay out of protocol crates.

## M6 — Facade Surface And Family Taxonomy Finalization

- [x] FCAB-120 [owner=codex] [deps=FCAB-030,FCAB-070,FCAB-090] [scope=siumai/src/lib.rs,siumai/src/provider_ext,siumai/src/{text,embedding,image,rerank,speech,transcription,video}.rs,docs/architecture,docs/migration]
  Goal: Finalize the public facade surface: stable prelude, provider extensions, protocol paths,
  `compat`, and `experimental`. Remove or demote broad glob mirrors that encourage cross-layer
  imports.
  Validation:
  - `cargo nextest run -p siumai --all-features --test public_surface_imports_test --no-fail-fast`
  - focused compile guards for touched provider extensions.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-120-review.md`.
  Evidence: provider-utils-backed facade root exports, narrowed unified prelude, explicit
  experimental modules, public-surface tests, migration notes, and `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-130 should now settle family taxonomy; retained broad exports are justified in
  `docs/architecture/public-surface.md`.

- [x] FCAB-130 [owner=codex] [deps=FCAB-120] [scope=docs/architecture,siumai-core/src/{video.rs,traits/video.rs,traits/music.rs},siumai-registry/src/registry/entry/handles,siumai/src/video.rs]
  Goal: Finalize family taxonomy. Treat video as a stable family if current source/docs support it;
  keep music extension-only unless an ADR is opened.
  Validation:
  - video family contract tests.
  - docs consistency checks for stable family list.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-130-review.md`.
  Evidence: seven-family architecture/ADR/migration docs, facade taxonomy guard, video family
  contract tests, music extension-only guard, and `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-140 should now run integration gates; if music becomes a candidate family later,
  split a follow-on instead of widening this task.

## M7 — Integration, Verification, And Closeout

- [x] FCAB-140 [owner=codex] [deps=FCAB-040,FCAB-060,FCAB-080,FCAB-100,FCAB-110,FCAB-130] [scope=docs,workspace]
  Goal: Run integration gates, update migration docs/changelog notes as needed, and ensure all
  source guards describe the final seams.
  Validation:
  - `cargo fmt --all --check` or narrower documented package formatting gates if full workspace
    formatting is impractical.
  - `./scripts/test-smoke.sh` or equivalent nextest package matrix.
  - broader closeout gate selected in `EVIDENCE_AND_GATES.md`.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-140-review.md`.
  Evidence: final validation log in `EVIDENCE_AND_GATES.md`.
  Handoff: FCAB-150 owns final workstream closeout and any follow-on split decisions.

- [x] FCAB-150 [owner=planner] [deps=FCAB-140] [scope=docs/workstreams/fearless-clean-architecture-boundaries]
  Goal: Close the lane or explicitly split unresolved work.
  Validation:
  - `WORKSTREAM.json` status, `TODO.md`, `MILESTONES.md`, `HANDOFF.md`, and evidence docs agree.
  - `git diff --check -- docs/workstreams/fearless-clean-architecture-boundaries`.
  Review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-150-review.md`.
  Evidence: closeout entry in `EVIDENCE_AND_GATES.md`.
  Handoff: Workstream closed. Residual future candidates are documented as non-blocking follow-ons.
