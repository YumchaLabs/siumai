# Fearless Clean Architecture Boundaries — Handoff

Status: Closed
Last updated: 2026-05-21

## Current State

The workstream has completed its initial seam inventory/source-guard baseline, the registry factory
facet split, built-in provider factory projection cleanup, and the OpenAI-compatible M2 deepening.
`ProviderFactory` remains the source-compatible custom-provider trait, but registry execution now
has narrower facets for stable family construction, legacy generic-client compatibility, and
extension capabilities. FCAB-040 centralized built-in provider typed-client `Arc` projection
helpers. FCAB-050 moved OpenAI-compatible completion response conversion and SSE parser state into
`siumai-protocol-openai`, so `siumai-provider-openai-compatible` delegates protocol conversion
instead of owning a local completion streaming parser. FCAB-060 moved TogetherAI image runtime
ownership into `siumai-provider-togetherai`, leaving the registry factory to compose provider-owned
image/rerank clients and the shared OpenAI-compatible text/audio runtime. FCAB-070 added
directional content namespaces across spec/core/facade surfaces and moved tests/examples/extras that
still need legacy `ContentPart` to explicit compatibility imports. FCAB-080 moved high-value
production response-side legacy content construction for OpenAI-compatible chat, Anthropic streaming,
Cohere, and Ollama behind named `response_content` adapters and recorded remaining direct
appearances in `content-part-adapter-audit.md`. FCAB-090 introduced `siumai-provider-utils` and
moved URL, MIME, builder-default, and chat-request-normalization implementations out of
`siumai-core::utils`; old core utility modules are migration aliases while provider/protocol crates
use internal `crate::provider_utils` aliases. FCAB-100 deepened `siumai-provider-utils` into the
canonical home for the rest of the AI SDK-style spec-only helper set (`data`, `download`,
`headers`, `id`, JSON parse/instruction helpers, provider options/references, reasoning mapping,
runtime metadata, serial jobs, settings, UTF-8 decoding, and type validation). The remaining
core-owned utility implementations are classified: `cancel` is stable core runtime stream wiring and
`streaming_tool_call` is explicit compat over core stream parts. FCAB-110 moved Gemini
GenerateContent request JSON normalization into
`siumai-protocol-gemini::standards::gemini::request_bridge`; `siumai-bridge` now keeps only a thin
Gemini shim plus bridge reports, loss policy, hooks, lifecycle, customization, and dispatch.
FCAB-120 finalized the facade public-surface slice: facade root utility helpers are now backed by
`siumai-provider-utils`, `ToolNameMapping` moved to `siumai-provider-utils::standards` with
`siumai-core::standards` left as a compatibility alias, `prelude::unified` keeps only the narrow
AI SDK-style helper subset, the experimental grouped `siumai_core` mirror was replaced by named
advanced modules, and retained broad exports are documented as explicit namespaces only. FCAB-130
settled the family taxonomy for this release line: the stable family list is Language, Embedding,
Image, Rerank, Speech, Transcription, and task-oriented Video. Music remains extension-only through
`MusicGenerationCapability` / provider extensions and has no stable `MusicModel` or registry
`music_model(...)` handle without a future ADR.

## Active Task

- None. The FCAB workstream is closed.
- Last closed task: FCAB-150.
- Final review: `docs/workstreams/fearless-clean-architecture-boundaries/reviews/FCAB-150-review.md`.
- Final evidence: closeout entries in `EVIDENCE_AND_GATES.md`.

## Decisions Since Last Update

- Opened a new workstream instead of reopening older lanes because `docs/workstreams/INDEX.md`
  records existing relevant fearless lanes as closed or superseded.
- Chose a single umbrella lane because the desired outcome crosses registry, provider,
  protocol, core, bridge, facade, and docs seams.
- Chose FCAB prefix for task IDs.
- Chose FCAB-020 guard baseline as the first executable task; moving code before guard refresh would
  make regressions harder to localize.
- Completed `seam-inventory.md` as the FCAB source-of-truth for current seam leaks and parallelism.
- Added a registry guard proving `ProviderFactory` generic `LlmClient` methods stay explicit
  `compat_*` aliases until FCAB-030/040 split them more deeply.
- Added a facade guard proving the FCAB seam inventory stays discoverable from the architecture
  guard suite.
- Completed FCAB-030 by adding `ProviderFamilyFactory`, `ProviderCompatibilityFactory`, and
  `ProviderExtensionFactory` facets.
- Registry handles now call `build_*` facet methods instead of the wide `ProviderFactory` methods
  for stable family and extension execution. Compatibility-only SiumaiBuilder construction uses the
  compatibility facet.
- Updated `docs/architecture/registry-without-builtins.md`, `docs/architecture/public-surface.md`,
  and `seam-inventory.md` to explain the facet split.
- Completed FCAB-040 by adding local typed-client projection helpers across the built-in provider
  factory set.
- OpenAI, generic OpenAI-compatible, DeepInfra, Fireworks, TogetherAI, Azure, Anthropic, Gemini,
  Google Vertex, Google Vertex MaaS, Google Vertex xAI, Groq, xAI, DeepSeek, MiniMaxi, Ollama,
  Cohere, Bedrock, and Anthropic Vertex factories now share local typed-client `Arc` projection
  helpers from their family and explicit compatibility methods.
- Added source guards:
  - `openai_compatible_factory_centralizes_checked_family_projection_glue`
  - `openai_factory_centralizes_family_projection_glue`
  - `promoted_openai_compatible_vendor_factories_centralize_projection_glue`
  - `builtin_provider_factories_centralize_typed_client_arc_projection`
- Completed FCAB-050 by adding protocol-owned OpenAI-compatible completion conversion in
  `siumai-protocol-openai::standards::openai::compat::completion`.
- Deleted the provider runtime's local OpenAI-compatible `completion/streaming.rs` parser module.
- `siumai-provider-openai-compatible` now delegates completion response and stream conversion through
  `CompletionResponseConversion` and `CompletionSseConverter`.
- Added source guard `openai_compatible_completion_streaming_conversion_is_protocol_owned`.
- Native OpenAI completion conversion remains provider-specific by design for this slice.
- Completed FCAB-060 by moving TogetherAI image request/response mapping, provider-option merging,
  image edit validation, response parsing, and HTTP execution into
  `siumai-provider-togetherai::providers::togetherai::image`.
- Added reusable TogetherAI JSON header construction in
  `siumai-provider-togetherai::standards::togetherai::headers` and reused it from rerank and image
  specs.
- `siumai-registry/src/registry/factories/togetherai.rs` now imports
  `TogetherAiImageClient` from the provider crate and only builds/projects it; the registry no
  longer owns `TogetherAiImageSpec`, `build_generation_body`, `build_edit_body`, image response
  parsing, or direct JSON execution for TogetherAI image.
- `TogetherAiBuilder::build_image_model` and `siumai::provider_ext::togetherai::TogetherAiImageClient`
  expose the provider-owned image client through the public provider extension surface.
- Added source guard `togetherai_provider_crate_owns_image_runtime` and extended
  `promoted_openai_compatible_vendor_factories_centralize_projection_glue` for the sync
  provider-owned TogetherAI image projection helper.
- Completed FCAB-070 by adding `content::prompt`, `content::output`, and `content::compat` namespaces in `siumai-spec::types`, inherited by `siumai-core::types`, and exposed from the `siumai` facade as `siumai::content::{prompt, output, compat}`.
- `siumai::prelude::unified` now exposes `prompt` and `output` navigation modules only; legacy `ContentPart` remains explicit under `siumai::compat::content`, `siumai::content::compat`, and `prelude::compat::content`.
- Updated public-surface and migration docs to teach directional content imports and explicit legacy compatibility imports.
- Added facade guards `directional_content_namespaces_are_visible_and_compat_is_explicit` and `tests_and_examples_do_not_import_legacy_content_part_from_unified_prelude`.
- Repointed tests, examples, and `siumai-extras` production/test code that still needs legacy `ContentPart` to `siumai::compat::content::ContentPart`.
- Completed FCAB-080 by moving OpenAI-compatible chat, Anthropic streaming, Cohere, and Ollama
  response-side legacy content construction behind local `response_content` adapters.
- Added adapter guards for those response paths and kept request `provider_options` out of response
  construction except empty legacy defaults.
- Recorded remaining direct `ContentPart` paths in `content-part-adapter-audit.md` as request-side
  serialization, bridge inspection/stream shims, metadata views, test-only fixtures, or later
  FCAB-090/100/110 follow-up work.
- Completed FCAB-090 by adding `siumai-provider-utils` to the workspace and moving
  `builder_helpers`, `chat_request`, `mime`, and `url` helper implementations there.
- `siumai-provider-utils` depends on `siumai-spec`, not `siumai-core`, so `siumai-core` can retain
  temporary compatibility aliases without creating a dependency cycle.
- Provider, protocol, and registry crates now depend on `siumai-provider-utils` and expose it
  internally as `crate::provider_utils`; high-churn helper call sites were rewired away from
  `crate::utils::{builder_helpers, chat_request, mime, url}`.
- Added source guard `provider_utils_crate_owns_high_churn_provider_helpers`.
- Completed FCAB-100 by moving the remaining spec-only AI SDK-style utility implementations into
  `siumai-provider-utils`: `data`, `download`, `error_message`, `headers`, `id`,
  `json_instruction`, `json_parse`, `option`, `provider_options`, `provider_reference`,
  `reasoning`, `runtime`, `serial_job`, `settings`, `utf8_decoder`, and `validate_types`.
- Matching `siumai-core::utils::*` modules are compatibility aliases backed by
  `siumai-provider-utils`; the source guard now covers all moved modules.
- The last stale provider import (`siumai-provider-togetherai` config API-key resolution) now uses
  `crate::provider_utils::builder_helpers`.
- Added `provider-utils-classification.md`, `core_utils_remaining_owned_modules_are_classified`,
  and `provider_protocol_crates_do_not_import_moved_provider_utils_from_core`.
- `siumai-core::utils::cancel` stays stable core runtime; `streaming_tool_call` stays explicit
  compat.
- Completed FCAB-110 by moving Gemini GenerateContent request JSON normalization into
  `siumai-protocol-gemini::standards::gemini::request_bridge`.
- `siumai-bridge/src/request/normalize/gemini_generate_content.rs` is now a thin compatibility shim
  that delegates to the protocol adapter; bridge-owned reports, hooks, loss policy, lifecycle,
  customization, and dispatch remain in `siumai-bridge`.
- Added `bridge-target-adapter-audit.md` and refreshed source guards
  `gemini_generate_content_request_normalization_is_protocol_adapter_backed` and
  `gemini_request_normalization_source_uses_provider_options_for_thought_signature`.
- Retained OpenAI/Anthropic direct-pair and normalization code is documented as bridge-owned until
  pure protocol parsing can be separated from bridge loss/replay policy.
- Completed FCAB-120 by moving `ToolNameMapping` implementation to
  `siumai-provider-utils::standards` and keeping `siumai-core::standards::tool_name_mapping` as a
  compatibility re-export.
- `siumai/src/lib.rs` now re-exports root utility helpers from `siumai-provider-utils`; the only
  explicit root `siumai-core::utils` exception is `delay` / `is_abort_error` until their
  cancellation-handle coupling is split.
- `siumai::prelude::unified` no longer exports broad provider-utils helper groups such as
  `Arrayable`, nullability helpers, base64/data helpers, reasoning mapping helpers, runtime
  user-agent/version helpers, URL helpers, media helpers, and low-level download/header/settings/JSON
  helpers. Those remain explicit root imports.
- `siumai::experimental::{defaults,execution,observability,params,retry,utils}` are now named
  advanced modules instead of a grouped `pub use siumai_core::{...}` mirror.
- Public-surface and migration docs now justify retained broad exports only under explicit
  namespaces: protocol, hosted tools, directional content, compatibility prelude, and experimental.
- Broader `cargo check -p siumai --all-features --tests` exposed stale implicit legacy
  `ContentPart` test imports; they now use explicit compatibility imports or existing local compat
  imports.
- Completed FCAB-130 by promoting Video to the documented stable family taxonomy as the seventh
  family while keeping the stable Rust video API task-oriented (`VideoModel`, `VideoModelV4`,
  `siumai::video::*`, and `VideoModelHandle`).
- Added `family_taxonomy_documents_video_as_stable_and_music_as_extension_only` to guard facade
  comments, public-surface docs, capability docs, module-split docs, ADR 0006, migration docs, core
  `VideoModel`, music capability-only status, registry `video_model(...)`, and absence of
  `music_model(...)`.
- Updated ADR 0001 and ADR 0006 with FCAB-130 taxonomy amendments: Music must not grow a stable
  `MusicModel` or registry `music_model(...)` without a future ADR.
- Verified registry video family contract tests and music extension delegation.
- Completed FCAB-140 integration/verification.
- `cargo fmt --all -- --check` is not usable on this Windows workspace because it fails with
  `文件名或扩展名太长。 (os error 206)`; FCAB-140 used documented per-package `cargo fmt --package
  ... -- --check` coverage instead.
- `bash ./scripts/test-smoke.sh` / `bash ./scripts/check-provider-deps.sh` are not usable in this
  terminal because `bash` resolves to `C:\Windows\system32\bash.exe` / WSL and hangs before script
  execution. FCAB-140 used a PowerShell-equivalent provider dependency scan plus an explicit nextest
  package matrix as the smoke substitute.
- `scripts/check-provider-deps.sh` now allows `siumai-provider-utils` as the intentional shared
  provider/protocol utility seam.
- FCAB-140 refreshed the facade architecture guards so the FCAB adapter audit participates in the
  production `ContentPart` coverage check and so the historical `prelude::registry` mirror assertion
  only inspects the compatibility prelude.
- Fresh integration gates passed for core, provider-utils, bridge, facade architecture guards,
  public surface imports, OpenAI protocol/provider packages, Anthropic/Gemini protocol packages,
  the focused registry feature matrix, and `siumai-registry --features all-providers`.
- Completed FCAB-150 closeout.
- `WORKSTREAM.json`, `TODO.md`, `MILESTONES.md`, `DESIGN.md`, `EVIDENCE_AND_GATES.md`, and this
  handoff now agree that the lane is closed.
- No immediate child workstream was opened. Residual items are deferred as concrete future
  candidates rather than blockers.

## Blockers

- No blocker remains for this lane.
- Residual operational note: bash smoke scripts are unreliable in the current PowerShell/WSL setup.
  FCAB-140 recorded a PowerShell-equivalent provider dependency scan plus an explicit nextest
  package matrix as the smoke substitute. Keep that matrix as the source of truth unless the local
  bash/WSL environment is fixed.

## Next Recommended Action

1. Review the final diff and commit after maintainer confirmation.
2. Use a Conventional Commit message such as
   `refactor: harden clean architecture boundaries`.
3. Open a new narrow workstream only for a concrete follow-on:
   - post-migration deletion of `siumai-core::utils::*` compatibility aliases;
   - moving remaining OpenAI/Anthropic bridge-owned direct-pair adapters once policy can stay in
     `siumai-bridge`;
   - promoting Music to a stable family via a dedicated ADR;
   - fixing local Windows bash/WSL smoke-script execution.
