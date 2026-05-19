# Fearless Module Deepening — Evidence And Gates

Status: Active
Last updated: 2026-05-19

## Smallest Current Proof

The first executable proof is provider-defined tool catalog extraction:

```bash
cargo nextest run -p siumai-spec --no-default-features
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses
cargo nextest run -p siumai-protocol-anthropic --features anthropic-standard
cargo nextest run -p siumai-protocol-gemini --features google
```

This proves `siumai-spec` still owns the passive tool data shape while protocol/provider-owned
Modules own concrete hosted tool catalogs.

## Gate Set

### Formatting Gate

Run package-scoped formatting for modified crates:

```bash
cargo fmt -p siumai-spec -p siumai-core -p siumai-protocol-openai -p siumai-protocol-anthropic -p siumai-protocol-gemini -p siumai -p siumai-registry -p siumai-bridge
```

Narrow the package list to touched crates when appropriate.

### Targeted Iteration Gates

#### Spec/provider residue

```bash
cargo nextest run -p siumai-spec --no-default-features
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses
cargo nextest run -p siumai-protocol-anthropic --features anthropic-standard
cargo nextest run -p siumai-protocol-gemini --features google
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast
```

#### Registry/family compatibility isolation

```bash
cargo nextest run -p siumai-registry --no-default-features
cargo nextest run -p siumai-registry --features builtins --no-default-features
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

#### Bridge/protocol deepening

```bash
cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google --test request_direct_bridge_fixtures_alignment_test --no-fail-fast
```

#### Content directional adapter proof

```bash
cargo nextest run -p siumai-spec --no-default-features prompt
cargo nextest run -p siumai-bridge --features openai,anthropic,google request
```

### Broader Closeout Gate

Prefer targeted crate gates because the workspace is large and provider features are combinatorial.
For final closeout, run at minimum:

```bash
cargo nextest run -p siumai-spec --no-default-features
cargo nextest run -p siumai-core --no-default-features
cargo nextest run -p siumai-registry --features builtins --no-default-features
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast
cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq --test public_surface_imports_test --no-fail-fast
```

If a slice touches all providers, add provider-specific package gates listed in that task.

### Review Gate

Run `review-workstream` before accepting a task or lane completion. Record blocking findings,
missing gates, and residual risks here or link to a journal note.

### Verification Gate

Run `verify-rust-workstream` before marking any task, Codex goal, or this lane complete. Fresh
verification is required even when earlier development gates passed.

## Evidence Anchors

- `docs/workstreams/fearless-module-deepening/DESIGN.md`
- `docs/workstreams/fearless-module-deepening/TODO.md`
- `docs/workstreams/fearless-module-deepening/MILESTONES.md`
- `docs/workstreams/fearless-module-deepening/HANDOFF.md`
- `docs/adr/0001-vercel-aligned-modular-split.md`
- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `repo-ref/ai/packages/provider`
- `repo-ref/ai/packages/provider-utils`
- `repo-ref/ai/packages/*/src/tool`

## Notes

- Do not claim full workspace health from a narrow gate.
- Do not mark a task complete if its architecture docs or source guards lag behind behavior.
- If a task reveals a larger public compatibility break, split a child workstream rather than
  silently widening this lane.

## Task Evidence Log

### 2026-05-19 — FMD-010 scope docs verification

Claim verified: the workstream has a complete planning baseline and is ready to move from
FMD-010 to FMD-020.

Commands:

```powershell
Get-ChildItem docs/workstreams/fearless-module-deepening -Force
Get-Content docs/workstreams/fearless-module-deepening/WORKSTREAM.json -Raw |
  ConvertFrom-Json |
  Select-Object title,status,active_task,next_task,updated
```

Result: PASS. The required planning files exist, `WORKSTREAM.json` parses, and the ledger now marks
FMD-010 complete with FMD-020 as the active implementation task.

Broader gates skipped: Rust build/test gates are not relevant to FMD-010 because this task only
created and reconciled workstream documentation.

### 2026-05-19 — FMD-020 provider-defined tool catalog extraction

Claim verified: canonical provider-defined hosted-tool catalogs no longer live in `siumai-spec` or
`siumai-core`; passive `Tool::ProviderDefined` data shapes remain in spec; protocol/provider crates
own concrete OpenAI, Anthropic, Google/Gemini, Groq, and xAI catalog helpers; facade compatibility
continues through `siumai::tools::*` re-exports.

Red evidence:

```powershell
cargo nextest run -p siumai-spec --no-default-features `
  spec_does_not_own_provider_defined_tool_catalogs `
  provider_defined_tool_data_surface_remains_passive
```

Result: EXPECTED FAIL before extraction. The new source guard observed that `siumai-spec/src/tools.rs`
and root `crate::tools::*` catalog ownership still existed.

Implementation summary:

- Deleted `siumai-spec/src/tools.rs` and `siumai-core/src/tools.rs`.
- Removed root `pub mod tools` ownership from `siumai-spec` and `siumai-core`, while keeping the
  passive `siumai-spec::types::tools` data module.
- Removed config-driven catalog helpers from spec data carriers:
  `Tool::provider_defined_id` and `ProviderDefinedTool::from_id`.
- Added protocol/provider-owned catalogs:
  - `siumai-protocol-openai/src/tool_catalog.rs`
  - `siumai-protocol-anthropic/src/tool_catalog.rs`
  - `siumai-protocol-gemini/src/tool_catalog.rs`
  - `siumai-provider-groq/src/tools.rs`
  - `siumai-provider-xai/src/tools.rs`
- Updated protocol/provider/bridge/facade users to import concrete hosted-tool facts from the
  owning protocol/provider crates.
- Kept `siumai::tools::*` as a compatibility facade that re-exports provider-owned catalogs and
  provides a facade-only aggregate `provider_defined_tool(id)` helper.

Fresh verification commands:

```powershell
cargo fmt --check -p siumai-spec -p siumai-core -p siumai-protocol-openai `
  -p siumai-protocol-anthropic -p siumai-protocol-gemini -p siumai-provider-openai `
  -p siumai-provider-anthropic -p siumai-provider-gemini -p siumai-provider-groq `
  -p siumai-provider-xai -p siumai-provider-openai-compatible `
  -p siumai-provider-google-vertex -p siumai-bridge -p siumai
cargo nextest run -p siumai-spec --no-default-features
cargo nextest run -p siumai-core --no-default-features core_does_not_own_provider_hosted_tool_factories
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses
cargo nextest run -p siumai-protocol-anthropic --features anthropic-standard
cargo nextest run -p siumai-protocol-gemini --features google
cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test public_surface_imports_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test facade_architecture_boundary_test --no-fail-fast
cargo nextest run -p siumai-provider-gemini --features google `
  google_interactions_request_warns_for_unsupported_provider_tool
cargo nextest run -p siumai-provider-groq --features groq
cargo nextest run -p siumai-provider-xai --features xai
```

Result: PASS.

Observed counts:

- `siumai-spec`: 262 passed.
- `siumai-core` targeted boundary guard: 1 passed.
- `siumai-protocol-openai`: 456 passed.
- `siumai-protocol-anthropic`: 219 passed.
- `siumai-protocol-gemini`: 138 passed.
- `siumai-bridge`: 108 passed.
- `siumai` public surface import guard: 25 passed.
- `siumai` facade architecture boundary guard: 24 passed.
- `siumai-provider-gemini` targeted unsupported-provider-tool regression: 1 passed.
- `siumai-provider-groq`: 55 passed.
- `siumai-provider-xai`: 67 passed.

Source guard:

```powershell
rg -n "siumai_core::tools|siumai_spec::tools|provider_defined_id|ProviderDefinedTool::from_id" . `
  -g "*.rs" --glob "!target/**" --glob "!repo-ref/**" --glob "!_third_party/**"
```

Result: PASS. Only the facade architecture boundary test assertion text references
`siumai_core::tools`.

Review: no blocking findings. Scope widened from the initial TODO list to provider Groq/xAI and
Google Vertex call sites because the old catalog had provider-owned Groq/xAI facts and because
downstream imports needed to follow the new ownership seam. This remains within FMD-020's
provider-residue-removal intent.

Broader gates skipped: full `cargo nextest run --workspace` was not run because the workspace has
large provider-feature combinatorics; FMD-020 was verified with all touched core/spec/protocol/bridge
facade gates plus the directly affected Groq/xAI provider gates.

### 2026-05-19 — FMD-030 provider-id-first classification seam

Claim verified: primary core and registry provider identity is provider-id-first; closed
`ProviderType` is retained only as legacy compatibility classification rather than the primary
client/catalog lookup path.

Red evidence:

```powershell
cargo nextest run -p siumai-core --no-default-features `
  core_primary_client_identity_is_provider_id_first
cargo nextest run -p siumai-registry --no-default-features `
  provider_catalog_lookup_is_provider_id_first
cargo nextest run -p siumai-spec --no-default-features `
  spec_provider_type_is_documented_as_compatibility_only
```

Result: EXPECTED FAIL before the slice. The new guards observed `LlmClient::provider_type`,
`ClientWrapper::provider_type`, registry catalog lookup through `ProviderType::from_name`, and
spec docs presenting `ProviderType` as a primary provider identity.

Implementation summary:

- Removed `provider_type()` from the primary `LlmClient` trait and `ClientWrapper` inherent API;
  `provider_id()` is now the primary client identity.
- Removed the unused `ProviderParamsExt` trait from core params.
- Added provider-id-first parameter validation entry points and report fields:
  `validate_for_provider_id`, `check_cross_provider_compatibility_by_id`,
  `optimize_for_provider_id`, plus `provider_id`/`source_provider_id`/`target_provider_id`
  report fields.
- Kept `validate_for_provider(...)`, `check_cross_provider_compatibility(...)`, and
  `optimize_for_provider(...)` as compatibility wrappers that convert `ProviderType` to ids.
- Introduced `siumai-registry::provider::legacy` as the narrow `ProviderType::from_name` owner.
- Added registry-owned `CatalogProviderId` classification for built-in catalog metadata so
  `provider_catalog` does not switch or look up through the public closed enum.
- Added `ProviderInfo::provider_id` and changed `get_provider_info_by_id` /
  `is_model_supported_by_id` to resolve ids/aliases directly.
- Kept `ProviderInfo::provider_type` and `ProviderMetadata::provider_type` as compatibility fields.
- Updated spec docs and guards to describe `ProviderType` as legacy compatibility classification
  and provider ids as open strings.

Fresh verification commands:

```powershell
cargo fmt --check -p siumai-spec -p siumai-core -p siumai-registry
cargo nextest run -p siumai-spec --no-default-features provider
cargo nextest run -p siumai-core --no-default-features --test core_provider_boundary_test
cargo nextest run -p siumai-registry --no-default-features
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
cargo nextest run -p siumai-registry --features openai --no-default-features `
  provider_catalog_lookup_by_id_maps_openai_family_variants `
  provider_catalog_keeps_custom_openai_compatible_variants `
  provider_catalog_lookup_by_id_works_for_openai_compatible
cargo nextest run -p siumai-registry --features azure --no-default-features `
  provider_catalog_lookup_by_id_maps_azure_chat_to_azure_family
```

Result: PASS.

Observed counts:

- `siumai-spec --no-default-features provider`: 84 passed.
- `siumai-core --no-default-features --test core_provider_boundary_test`: 40 passed.
- `siumai-registry --no-default-features`: 98 passed.
- `siumai --test facade_architecture_boundary_test --no-fail-fast`: 24 passed.
- `siumai-registry --features openai --no-default-features` targeted catalog variants: 3 passed.
- `siumai-registry --features azure --no-default-features` targeted Azure alias: 1 passed.

Source guard:

```powershell
rg -n "ProviderType::from_name|fn provider_type\(|pub fn provider_type\(|ProviderParamsExt|\
get_provider_info\(&ProviderType::from_name|is_model_supported\(&ProviderType::from_name|\
let ptype = ProviderType::from_name" `
  siumai-core/src siumai-registry/src siumai-spec/src siumai/src
```

Result: PASS. Production hits are limited to `siumai-registry/src/provider/legacy.rs` and
`siumai-spec/src/types/common.rs` compatibility tests. Primary core and registry catalog paths no
longer route through `ProviderType::from_name`.

Review: no blocking findings. This slice intentionally does not remove the public `ProviderType`
enum or compatibility fields because that would be a downstream breaking change; FMD-040/FMD-050
should continue isolating compatibility modules rather than widening this task into public removal.

Broader gates skipped: full workspace/all-provider testing was not run because FMD-030 changed
provider identity seams in `siumai-spec`, `siumai-core`, `siumai-registry`, and facade boundary
tests only; targeted no-default plus OpenAI/Azure alias catalog gates cover the changed behavior.

### 2026-05-19 — FMD-040 generic-client compatibility isolation

Claim verified: the physical `LlmClient` / `ClientWrapper` implementation is now scoped under an
explicit compatibility module, stable registry family handles remain family-first, and public
migration paths are documented for downstream generic-client imports.

Red evidence:

```powershell
cargo nextest run -p siumai-core --no-default-features --test core_provider_boundary_test `
  llm_client_is_physically_scoped_under_compat_module
cargo nextest run -p siumai-registry --no-default-features `
  registry_generic_client_imports_are_compat_scoped `
  registry_root_does_not_mirror_broad_core_modules
cargo nextest run -p siumai --test facade_architecture_boundary_test `
  facade_generic_client_paths_are_explicit_compatibility_exports --no-fail-fast
```

Result: EXPECTED FAIL before isolation. The new guards observed that
`siumai_core::compat::client` / `siumai::compat::client` / `siumai_registry::compat::client` did not
exist and that registry production code still imported generic clients through broad `client` roots.

Implementation summary:

- Added `siumai-core/src/compat/client.rs` as the physical home for `LlmClient`,
  `ClientWrapper`, generic-client capability discovery, and downcast helpers.
- Kept `siumai-core/src/client.rs` as a lower-level migration alias only.
- Added explicit facade and registry compatibility imports:
  `siumai::compat::client::{LlmClient, ClientWrapper}` and
  `siumai_registry::compat::client::{LlmClient, ClientWrapper}`.
- Removed the registry root `siumai_registry::LlmClient` export and documented the replacement
  `siumai_registry::compat::client::LlmClient` path.
- Updated registry, facade, provider crates, and examples to import generic clients through explicit
  compatibility modules instead of broad `crate::client` or `siumai_core::client` roots.
- Added source guards for core physical ownership, registry production import paths, facade
  compatibility exports, and public migration docs.

Review:

- Blocking finding fixed during review: removing the registry root `LlmClient` alias needed an
  explicit migration note. `docs/architecture/public-surface.md`,
  `docs/migration/migration-0.11.0-beta.7.md`, and
  `public_docs_classify_generic_llm_client_factory_paths_as_migration_only` now guard that path.
- No remaining blocking code-quality findings. The provider-crate import updates are scope widening,
  but they were necessary to prevent local provider roots from keeping broad core `client` aliases.

Fresh verification commands:

```powershell
cargo fmt --check -p siumai-core -p siumai-registry -p siumai `
  -p siumai-provider-openai -p siumai-provider-anthropic -p siumai-provider-gemini `
  -p siumai-provider-openai-compatible -p siumai-provider-google-vertex `
  -p siumai-provider-groq -p siumai-provider-xai -p siumai-provider-azure `
  -p siumai-provider-amazon-bedrock -p siumai-provider-cohere `
  -p siumai-provider-deepseek -p siumai-provider-minimaxi `
  -p siumai-provider-ollama -p siumai-provider-togetherai
rg -n "crate::client::|siumai_core::client|use crate::client|use siumai_core::client|\
siumai_registry::LlmClient|use siumai_registry::LlmClient" . -g "*.rs" `
  --glob "!target/**" --glob "!repo-ref/**" --glob "!_third_party/**" --glob "!**/tests/**" -S
cargo nextest run -p siumai-core --no-default-features --test core_provider_boundary_test
cargo nextest run -p siumai-registry --no-default-features
cargo nextest run -p siumai-registry --features builtins --no-default-features
cargo nextest run -p siumai --test facade_architecture_boundary_test --no-fail-fast
```

Result: PASS.

Observed counts:

- `siumai-core --no-default-features --test core_provider_boundary_test`: 41 passed.
- `siumai-registry --no-default-features`: 99 passed.
- `siumai-registry --features builtins --no-default-features`: 104 passed.
- `siumai --test facade_architecture_boundary_test --no-fail-fast`: 25 passed.
- Production source guard excluding tests: no matches.

Provider import validation:

```powershell
cargo nextest run -p siumai-provider-amazon-bedrock --features bedrock --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-anthropic --features anthropic --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-azure --features azure --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-cohere --features cohere --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-deepseek --features deepseek --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-gemini --features google --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-google-vertex --features google-vertex --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-groq --features groq --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-minimaxi --features minimaxi --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-ollama --features ollama --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-openai --features openai --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-openai-compatible --features openai-standard --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-togetherai --features togetherai --no-fail-fast --no-tests pass
cargo nextest run -p siumai-provider-xai --features xai --no-fail-fast --no-tests pass
```

Result: PASS. All affected provider packages compiled and ran their feature-selected nextest suites
successfully under the listed feature sets.

Retry note: an initial provider matrix command included a stale `provider_id` feature for
`siumai-provider-amazon-bedrock` and failed before code execution. The matrix was rerun with the
current package feature names listed above and passed.

Broader gates skipped: full `cargo nextest run --workspace --all-features` was not run because the
workspace has large provider feature combinatorics. FMD-040 was verified with the required
registry/facade gates, the core ownership boundary guard, a production source guard, formatting, and
all touched provider package feature gates.

### 2026-05-19 — FMD-050 registry broad build-helper isolation

Claim verified: production built-in registry factories no longer depend on the old broad
`registry::factory::build_*_client(...)` generic-client helpers. Those public helpers are now
explicitly deprecated compatibility shims; new provider construction goes through provider-owned
private typed builders and `ProviderFactory::*_family_with_ctx(...)` methods.

Red evidence:

```powershell
cargo nextest run -p siumai-registry --features all-providers --no-default-features `
  production_factories_do_not_call_legacy_broad_build_client_helpers `
  legacy_registry_factory_build_helpers_are_deprecated_compatibility_shims --no-fail-fast
```

Result: EXPECTED FAIL before the slice. The new guards observed:

- `openai.rs` still called `crate::registry::factory::build_openai_client(...)`.
- `openai.rs` still called `crate::registry::factory::build_openai_chat_completions_client(...)`.
- `anthropic.rs` still called `crate::registry::factory::build_anthropic_client(...)`.
- `registry::factory` docs did not classify the broad helpers as compatibility-only and the public
  broad helpers were not deprecated.

Implementation summary:

- Changed `OpenAIProviderFactory::compat_language_client_with_ctx(...)` to reuse its private
  `build_family_model_with_ctx(...)` typed builder instead of calling broad registry factory
  helpers.
- Added `AnthropicProviderFactory::build_text_family_model_with_ctx(...)` and routed both
  `compat_language_client_with_ctx(...)` and `language_model_text_with_ctx(...)` through it.
- Marked the remaining public broad `registry::factory::build_*_client(...)` functions as
  deprecated compatibility shims and documented that new code should use provider config-first
  construction inside family-first `ProviderFactory` methods.
- Left `Option<()>` placeholder parameters on those deprecated shims to avoid silently changing an
  already-public compatibility signature in this slice.
- Added architecture guards for:
  - production factories not calling broad legacy build helpers,
  - legacy broad helpers being compatibility-only/deprecated,
  - `all-providers` including `google-vertex`.
- Added `google-vertex` to `siumai-registry`'s `all-providers` feature because the built-in
  OpenAI-compatible catalog includes Google Vertex xAI entries that require the Google Vertex
  factory.

Review:

- No blocking findings after fixes. The only scope expansion was the `all-providers` feature list:
  the advertised FMD-050 validation gate failed until Google Vertex was included, so this was
  necessary to make the gate match the provider catalog's actual contents.

Fresh verification commands:

```powershell
cargo fmt --check -p siumai-registry
rg -n "crate::registry::factory::build_(openai_client|openai_chat_completions_client|\
openai_compatible_client|anthropic_client|gemini_client|anthropic_vertex_client|\
google_vertex_client|ollama_client|minimaxi_client)\(" `
  siumai-registry/src/registry/factories siumai-registry/src/provider -g "*.rs"
cargo nextest run -p siumai-registry --features all-providers --no-default-features `
  all_providers_feature_includes_google_vertex_family `
  create_registry_with_defaults_registers_native_factories --no-fail-fast
cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast
```

Result: PASS.

Observed counts:

- Targeted Google Vertex/all-provider regression: 2 passed.
- Full FMD-050 registry gate:
  `cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast`
  — 536 passed.
- Production broad-helper source guard: no matches.
- Formatting: pass.

Retry note:

- The first full all-provider gate run failed with 487 passed / 1 failed because `all-providers`
  omitted `google-vertex` while the OpenAI-compatible catalog exposed `google-vertex-xai`. After
  adding `google-vertex` to `all-providers` and guarding that manifest invariant, the full gate
  passed with 536 tests.

Broader gates skipped: full workspace testing was not run because FMD-050 is scoped to
`siumai-registry` factory construction; the task's advertised all-provider registry gate plus source
guards cover the changed behavior.

### 2026-05-19 — FMD-060 bridge request normalization adapter split

Claim verified: Gemini GenerateContent request normalization is no longer owned by the monolithic
`siumai-bridge/src/request/normalize.rs` parser body. The public Gemini bridge entry points still
live at the existing `request` API surface, but they now delegate to the narrow
`siumai-bridge/src/request/normalize/gemini_generate_content.rs` adapter, which owns the
Gemini-specific typed request parsing, content/part role policy, provider-defined tool mapping,
tool config mapping, and generation config preservation.

Red evidence:

```powershell
cargo nextest run -p siumai-bridge --features openai,anthropic,google `
  gemini_generate_content_request_normalization_is_protocol_adapter_backed --no-fail-fast
```

Result: EXPECTED FAIL before the adapter extraction. The new guard failed because
`src/request/normalize/gemini_generate_content.rs` did not exist and `normalize.rs` still owned the
Gemini typed parser.

Implementation summary:

- Added `siumai-bridge/src/request/normalize/gemini_generate_content.rs` as the Gemini
  GenerateContent JSON -> `ChatRequest` adapter.
- Moved the Gemini-specific typed parser and helper policy out of `request/normalize.rs`,
  including `GeminiGenerateContentRequest`, Gemini content/part role mapping, thought-signature
  provider options, tool-call id pairing, provider-defined Gemini tool mapping, `toolConfig`, and
  `generationConfig` handling.
- Kept `request/normalize.rs` as the stable public wrapper owner for
  `bridge_gemini_generate_content_json_to_chat_request*`, delegating to
  `gemini_generate_content::parse_json_to_chat_request(...)`.
- Repointed the existing thought-signature source guard to the new adapter file and added an
  adapter ownership source guard that prevents the Gemini typed parser and tool/content policy from
  returning to the monolithic normalizer.

Review:

- No blocking findings. The split is behavior-preserving and improves locality by moving a
  feature-gated, protocol-specific parser behind a narrow module seam while leaving shared request
  primitive helpers common.
- Residual risk: the adapter still depends on several `normalize.rs` shared helpers through
  `pub(super)` visibility. That is acceptable for this first FMD-060 slice, but future bridge
  deepening may want a dedicated shared request primitive module if OpenAI/Anthropic extractions
  reveal wider reuse pressure.

Fresh verification commands:

```powershell
cargo fmt --check -p siumai-bridge
cargo nextest run -p siumai-bridge --features openai,anthropic,google gemini --no-fail-fast
cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google `
  --test request_direct_bridge_fixtures_alignment_test --no-fail-fast
```

Result: PASS.

Observed counts:

- Targeted Gemini bridge filter: 7 passed.
- Full FMD-060 bridge gate:
  `cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast`
  — 109 passed.
- Facade direct request bridge fixture alignment gate:
  `cargo nextest run -p siumai --features openai,anthropic,google --test request_direct_bridge_fixtures_alignment_test --no-fail-fast`
  — 7 passed.
- Formatting: pass.

Broader gates skipped: full workspace testing and the FMD-070 OpenAI protocol gate were not run
because FMD-060 only changed bridge request normalization plus source guards. The advertised bridge
and facade fixture gates passed and preserve request bridge semantics for this slice.

### 2026-05-19 — FMD-070 OpenAI protocol internal deepening

Claim verified: OpenAI Responses request mapping and response parsing now have narrower behavior
Modules instead of concentrating request body assembly, hosted/dynamic output item helpers, and
provider metadata/source/logprob aggregation in monolithic transformer files. The OpenAI protocol
all-features gate and targeted facade OpenAI fixture gates still pass.

Red evidence:

```powershell
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses `
  openai_responses_request_transformer_has_deep_request_builder_module --no-fail-fast
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses `
  openai_responses_response_transformer_has_hosted_tool_output_module --no-fail-fast
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses `
  openai_responses_response_transformer_has_metadata_module --no-fail-fast
```

Result: EXPECTED FAIL before each extraction. The guards respectively observed that:

- `request/responses.rs` still defined inline `ResponsesHooks` with `build_base_chat_body(...)` and
  `post_process_chat(...)`.
- `response/responses.rs` did not have a `hosted_tools` submodule and still owned hosted/dynamic
  output helper functions.
- `response/responses.rs` did not have a `metadata` submodule and still owned Responses provider
  metadata/source/logprobs aggregation.

Implementation summary:

- Added
  `siumai-protocol-openai/src/standards/openai/transformers/request/responses/responses_request_builder.rs`
  and moved Responses request base body construction plus typed provider option post-processing
  behind `ResponsesRequestHooks`.
- Added
  `siumai-protocol-openai/src/standards/openai/transformers/response/responses/hosted_tools.rs`
  and moved hosted/dynamic output item helpers for xAI file search queries, file-search results,
  local shell input, apply-patch input, and provider-executed shell environment detection.
- Added
  `siumai-protocol-openai/src/standards/openai/transformers/response/responses/metadata.rs`
  and moved response-level provider metadata aggregation, tool/message source collection,
  deduplication, custom provider metadata key wrapping, and output-text logprob extraction.
- Kept `extract_responses_output_text_logprobs` as a crate-visible compatibility alias for existing
  Responses SSE conversion code, while moving the implementation into the metadata module.
- Added source guards to keep these behavior seams from regressing back into monolithic transformer
  files.

Review:

- No blocking findings. The split follows the workstream rule to split by behavior seam rather than
  file size: request body assembly, hosted/dynamic output item mapping, and metadata/source/logprob
  aggregation now have narrow internal Interfaces.
- Streaming accumulation already has dedicated converter submodules (`state`, `sse`, `stream_meta`,
  `tool_events`, provider-tool-specific helpers, and serializers). This task validated stream
  behavior through protocol and facade fixture gates rather than adding a wrapper-only split.
- Residual risk: `responses_sse/converter/convert.rs` and `serialize.rs` remain large and should be
  revisited only when a concrete stream behavior seam is selected; file size alone is not a reason
  for another split.

Fresh verification commands:

```powershell
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses `
  openai_responses_response_transformer_has_metadata_module `
  responses_file_search_sources_roundtrip_with_tool_scoped_metadata `
  responses_transformer_keeps_distinct_file_citation_sources_per_index `
  responses_transformer_surfaces_typed_source_metadata_for_container_and_file_path `
  xai_responses_transformer_uses_xai_usage_semantics --no-fail-fast
cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses `
  responses_transformer --no-fail-fast
cargo fmt --check -p siumai-protocol-openai
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_chat_messages_fixtures_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_chat_completions_response_bridge_roundtrip_fixtures_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_chat_completions_stream_bridge_roundtrip_fixtures_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_responses_input_fixtures_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_responses_response_fixtures_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_responses_response_provider_metadata_key_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_responses_file_search_fixtures_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_responses_file_search_stream_alignment_test --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test openai_responses_stream_bridge_roundtrip_fixtures_alignment_test --no-fail-fast
```

Result: PASS.

Observed counts:

- Metadata/source targeted guard and semantic checks: 5 passed.
- Responses response transformer filter: 11 passed.
- Full FMD-070 protocol gate:
  `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast` — 461 passed.
- Facade OpenAI fixture gates:
  - chat messages request/body fixtures: 1 passed.
  - chat completions response bridge fixtures: 1 passed.
  - chat completions stream bridge fixtures: 3 passed.
  - Responses input fixtures: 1 passed.
  - Responses response fixtures: 1 passed.
  - Responses provider metadata key fixture: 1 passed.
  - Responses file-search tool fixtures: 1 passed.
  - Responses file-search stream fixture: 1 passed.
  - Responses stream bridge fixture: 1 passed.
- Formatting: pass.

Retry note: no code-related gate retries. Some `cargo` invocations printed package-cache file-lock
waits while other builds released locks, then completed successfully.

Broader gates skipped: full workspace testing was not run because FMD-070 is scoped to
`siumai-protocol-openai/src/standards/openai` plus targeted facade OpenAI fixtures. The required
protocol all-features gate, formatting, source guards, response/request behavior filters, and facade
fixture gates passed.

### 2026-05-19 — FMD-080 legacy ContentPart directional boundary decision

Claim verified: FMD-080 has a concrete decision and adapter migration plan. This lane will not open
the breaking `ContentPart` namespace-move child workstream yet; it will implement a non-breaking
request-side bridge legacy adapter extraction in FMD-090.

Decision artifact:

- `docs/workstreams/fearless-module-deepening/FMD-080-content-part-directional-boundary-decision.md`

Inputs reviewed:

- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- `docs/workstreams/fearless-content-part-boundary-split/`
- current `siumai-spec` prompt and generated-output projection code
- current `siumai-bridge/src/request/normalize.rs` request-side legacy adapter helpers
- current `siumai-core/src/streaming/processor.rs` response-side `response_text_part(...)` adapter
- current `siumai/src/text.rs` generate-text projection delegation to `siumai-spec`

Decision summary:

- Keep legacy `ContentPart` compatibility-only at existing public paths for this lane.
- Do not open the breaking namespace-move workstream yet, because ADR-0008's preconditions are not
  fully met and stable serde/protocol payloads still use `ContentPart` legitimately.
- Use FMD-090 for a non-breaking adapter-first proof slice: extract bridge request normalization's
  legacy `ContentPart` construction helpers into `siumai-bridge/src/request/legacy_content.rs`.
- Guard the new boundary so request-side compatibility construction keeps response
  `provider_metadata` empty and does not drift back into `request/normalize.rs`.

Review:

- Planner self-review found no blocking issues. The decision follows ADR-0008 and the closed
  `fearless-content-part-boundary-split` handoff: adapter-first, preserve public compatibility, and
  defer the breaking namespace move until directional adapters cover more main request and response
  paths.

Validation:

```powershell
Get-Content -Path 'docs/workstreams/fearless-module-deepening/WORKSTREAM.json' | ConvertFrom-Json | ConvertTo-Json -Depth 10
rg -n "FMD-080|active_task|next_task|legacy ContentPart" \
  docs/workstreams/fearless-module-deepening/WORKSTREAM.json \
  docs/workstreams/fearless-module-deepening/TODO.md \
  docs/workstreams/fearless-module-deepening/HANDOFF.md \
  docs/workstreams/fearless-module-deepening/FMD-080-content-part-directional-boundary-decision.md
```

Result: PASS. The workstream pointer now advances to FMD-090 / FMD-100, and the decision artifact is
present.

Broader gates skipped: no Rust code changed in FMD-080. The task validation is a design note or new
workstream with explicit request/response/compat target state, so source/test gates are deferred to
FMD-090 implementation.

### 2026-05-19 — FMD-090 request-side legacy ContentPart adapter extraction

Claim verified: bridge request normalization no longer owns the legacy request `ContentPart`
constructor helpers directly. The approved non-breaking proof slice now centralizes request-side
legacy compatibility construction in `siumai-bridge/src/request/legacy_content.rs`, routes
OpenAI/Anthropic/Gemini request normalization through that adapter, and keeps response
`provider_metadata` empty at the request boundary.

Red evidence:

```powershell
cargo nextest run -p siumai-bridge --features openai,anthropic,google `
  request_normalization_centralizes_legacy_request_content_constructors --no-fail-fast
```

Result: EXPECTED FAIL before extraction. The source guard failed at compile time because
`siumai-bridge/src/request/legacy_content.rs` did not exist yet.

Implementation summary:

- Added `siumai-bridge/src/request/legacy_content.rs` as the request-side adapter module for the
  legacy `ContentPart` compatibility carrier.
- Moved the request helper family into that module:
  `request_text_part`, `request_reasoning_part`, `request_image_part`, `request_audio_part`,
  `request_file_part`, `request_tool_call_part`, and `request_tool_result_part`.
- Updated OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini
  GenerateContent request normalization paths to construct request parts through
  `legacy_content::*` instead of direct `ContentPart::text`, `ContentPart::reasoning`,
  `ContentPart::tool_call`, or local helper definitions.
- Kept the plain-text collapse match in `message_from_parts(...)` as the only
  `provider_metadata: None` occurrence in `request/normalize.rs`; actual request construction now
  centralizes that invariant in `request/legacy_content.rs`.
- Added a source guard requiring `mod legacy_content;`, requiring adapter calls from the normalizer,
  forbidding the helper definitions in `request/normalize.rs`, and asserting every adapter-side
  `provider_metadata` assignment remains `None`.

Review:

- 2026-05-19 `review-workstream` self-review found no blocking workstream-compliance or
  code-quality issues.
- The scope intentionally stayed non-breaking: public `ContentPart` paths and serde shape were not
  renamed or removed.
- Residual risk: this is only a request-side adapter proof. Response-side generated-output
  projection and any breaking namespace move remain deferred to a future compatibility-break
  workstream.

Fresh verification commands:

```powershell
cargo nextest run -p siumai-bridge --features openai,anthropic,google `
  request_normalization_centralizes_legacy_request_content_constructors --no-fail-fast
cargo nextest run -p siumai-bridge --features openai,anthropic,google request --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google `
  --test request_direct_bridge_fixtures_alignment_test --no-fail-fast
cargo fmt --check -p siumai-bridge
```

Result: PASS.

Observed counts:

- Source guard:
  `request_normalization_centralizes_legacy_request_content_constructors` — 1 passed.
- Bridge request gate:
  `cargo nextest run -p siumai-bridge --features openai,anthropic,google request --no-fail-fast`
  — 48 passed, 61 skipped.
- Facade direct request bridge fixture alignment gate:
  `cargo nextest run -p siumai --features openai,anthropic,google --test request_direct_bridge_fixtures_alignment_test --no-fail-fast`
  — 7 passed.
- Formatting: pass.

Skipped gates:

- `cargo nextest run -p siumai-spec --no-default-features prompt --no-fail-fast` was not run
  because FMD-090 did not touch `siumai-spec` prompt projection helpers or facade text projection
  code.
- Full workspace testing was not run because FMD-090 is scoped to `siumai-bridge` request
  normalization plus the facade bridge fixture guard; the targeted bridge/facade gates cover the
  changed paths.

### 2026-05-19 — FMD-100 provider public-path parity test locality

Claim verified: the largest provider public-path parity test is no longer a single oversized
provider-change hotspot. The shared harness remains in
`siumai/tests/provider_public_path_parity_test.rs`, while provider-specific parity scenarios now
live in provider-local modules under `siumai/tests/provider_public_path_parity/`. The same test
binary name and public gate remain unchanged.

Red evidence:

```powershell
cargo nextest run -p siumai-registry --features all-providers --no-default-features `
  provider_public_path_parity_test_is_split_by_provider_module `
  migrated_public_path_modules_use_registry_builder_shortcuts `
  focused_public_facade_tests_use_registry_owned_builtin_factory_resolution --no-fail-fast
```

Result: EXPECTED FAIL during the first FMD-100 split iteration for
`migrated_public_path_modules_use_registry_builder_shortcuts`, because the new manifest was too
strict for `deepinfra_public_path`. DeepInfra legitimately uses the shared
`built_in_registry_builder(...)` helper rather than the provider-specific
`.with_provider_api_key_base_url_fetch(...)` shortcut in that module. The manifest was narrowed to
assert the actual expected locality marker for DeepInfra, then the guard passed.

Implementation summary:

- Split `siumai/tests/provider_public_path_parity_test.rs` into:
  - a shared test harness/root with imports, transports, fixture helpers, and `#[path = ...]`
    provider module declarations;
  - provider-local module files under `siumai/tests/provider_public_path_parity/` for OpenAI,
    Azure, Gemini, Cohere, TogetherAI, DeepInfra, Vertex MaaS, Google Vertex xAI, DeepSeek,
    OpenAI-compatible audio families, Groq, Ollama, MiniMaxi, Bedrock, Anthropic, Vertex
    Anthropic/Gemini, and xAI.
- Added registry architecture guards that:
  - require provider public-path parity scenarios to be split by provider module;
  - read provider module files through a manifest instead of brittle source-span slicing;
  - preserve the existing registry-owned factory/helper path checks across the split.
- Left `public_surface_imports_test.rs` broad because it is intentionally a compile-surface import
  guard and did not present the same provider-local edit pain as the 47k-line parity monolith.

Review:

- 2026-05-19 `review-workstream` self-review found no blocking workstream-compliance or
  code-quality issues.
- The split is behavior-preserving: all provider public-path parity test names remain under the
  same `provider_public_path_parity_test` test binary, but failures and edits now localize to a
  provider module file.
- Residual risk: several provider-local modules are still large because individual providers have
  deep parity coverage. Further splitting should be provider-specific and driven by concrete edit
  pain, not by file-size alone.

Fresh verification commands:

```powershell
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
cargo nextest run -p siumai --test provider_public_path_parity_test --no-fail-fast
cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast
cargo fmt --check -p siumai -p siumai-registry
```

Result: PASS.

Observed counts:

- Public surface import gate:
  `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast` — 20 passed.
- Provider public-path parity gate with default feature set:
  `cargo nextest run -p siumai --test provider_public_path_parity_test --no-fail-fast` —
  150 passed.
- Broader provider public-path parity regression with all relevant provider features:
  `cargo nextest run -p siumai --test provider_public_path_parity_test --features openai,anthropic,google,google-vertex,xai,groq,cohere,togetherai,deepinfra,bedrock,deepseek,ollama,minimaxi,azure --no-fail-fast`
  — 505 passed.
- Registry all-provider gate:
  `cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast`
  — 537 passed.
- Formatting: pass.

Retry note:

- A raw `rustfmt` command failed because it was invoked directly without Cargo edition context and
  because the PowerShell wildcard was passed literally. The repository-standard `cargo fmt -p
  siumai -p siumai-registry` command was used instead and succeeded.
- Some cargo invocations printed package-cache file-lock waits while other builds released locks,
  then completed successfully.

Broader gates skipped: full workspace testing was not run because FMD-100 changed test layout and
registry architecture guards only. The advertised public-surface/parity and registry all-provider
gates passed, plus an extra all-provider-feature run of the split parity test proved the provider
modules compile together.

### 2026-05-19 — FMD-110 closeout verification

Claim verified: the Fearless Module Deepening lane is complete. FMD-010 through FMD-110 are done,
the shipped module boundaries match the workstream target state, and the only remaining
public-shape break is explicitly deferred to a future compatibility-break `ContentPart` workstream.

Closeout review:

- `review-workstream` self-review found no blocking workstream-compliance or code-quality findings.
- `TODO.md` now has no unchecked tasks.
- `WORKSTREAM.json` parses and is marked `complete`.
- `HANDOFF.md` records the residual follow-on instead of leaving journal-only decisions.

Shipped seams:

- Provider-defined hosted-tool catalog ownership moved from `siumai-spec`/`siumai-core` into
  protocol/provider-owned catalog modules.
- Core and registry primary identity paths are provider-id-first; closed `ProviderType` remains
  compatibility classification.
- `LlmClient`/`ClientWrapper` and generic-client downcast paths are physically scoped under
  explicit `compat` modules.
- Registry production factories route through provider-owned typed builders and family-first
  `ProviderFactory` methods instead of broad legacy `build_*_client` helpers.
- Bridge request normalization has narrower adapter seams, including Gemini GenerateContent and the
  request-side legacy `ContentPart` adapter.
- OpenAI Responses request and response internals are split around request body assembly,
  hosted-tool output handling, and metadata/source/logprobs aggregation.
- Provider public-path parity scenarios now live in provider-local modules behind the same test
  binary.

Fresh closeout verification commands:

```powershell
cargo fmt --check -p siumai-spec -p siumai-core -p siumai-registry -p siumai-bridge `
  -p siumai-protocol-openai -p siumai-protocol-anthropic -p siumai-protocol-gemini -p siumai
Get-Content docs/workstreams/fearless-module-deepening/WORKSTREAM.json -Raw |
  ConvertFrom-Json | Select-Object title,status,active_task,next_task,updated | Format-List
Select-String -Path docs/workstreams/fearless-module-deepening/TODO.md -Pattern '^- \[ \]'
cargo nextest run -p siumai-spec --no-default-features --no-fail-fast
cargo nextest run -p siumai-core --no-default-features --test core_provider_boundary_test --no-fail-fast
cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast
cargo nextest run -p siumai --features openai,anthropic,google,xai,groq `
  --test public_surface_imports_test --no-fail-fast
cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast
cargo nextest run -p siumai --test provider_public_path_parity_test `
  --features openai,anthropic,google,google-vertex,xai,groq,cohere,togetherai,deepinfra,bedrock,deepseek,ollama,minimaxi,azure --no-fail-fast
```

Result: PASS.

Observed counts:

- Formatting gate: pass.
- Pre-closeout task pointer check: `WORKSTREAM.json` parsed; only FMD-110 was unchecked before
  this closeout update.
- Spec no-default-features gate: 263 passed.
- Core provider boundary gate: 41 passed.
- Bridge OpenAI/Anthropic/Google feature gate: 109 passed.
- Facade public surface gate with OpenAI/Anthropic/Google/xAI/Groq features: 25 passed.
- Registry all-provider gate:
  `cargo nextest run -p siumai-registry --features all-providers --no-default-features --no-fail-fast`
  — 537 passed.
- OpenAI protocol all-features gate:
  `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast` — 461 passed.
- Anthropic protocol all-features gate:
  `cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast` — 219 passed.
- Gemini protocol all-features gate:
  `cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast` — 138 passed.
- All-provider provider public-path parity gate:
  `cargo nextest run -p siumai --test provider_public_path_parity_test --features openai,anthropic,google,google-vertex,xai,groq,cohere,togetherai,deepinfra,bedrock,deepseek,ollama,minimaxi,azure --no-fail-fast`
  — 505 passed.

Retry note:

- Several Cargo invocations printed package-cache file-lock waits while other builds released locks,
  then completed successfully.
- No closeout command required code changes or destructive git operations.

Broader gates skipped:

- Full `cargo nextest run --workspace` was not run because this workstream spans many optional
  provider feature combinations. Closeout instead used fresh gates for every touched architectural
  stratum: spec, core boundary, bridge, registry all-provider, three protocol crates all-features,
  facade public surface, and all-provider provider public-path parity.

Deferred follow-on:

- The breaking public `ContentPart` namespace move / response-side generated-output adapter
  deepening remains out of this lane by design. Open a dedicated compatibility-break workstream for
  that scope after ADR-0008 preconditions are met.
