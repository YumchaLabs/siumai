# Native Extension And Compat Retirement — Evidence And Gates

Status: Active
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
rg -n "impl ProviderExtensionFactory|compat_.*_client_with_ctx|build_.*_extras_with_ctx|as_.*_extras" siumai-registry/src
```

This inventories the remaining extension factory and compatibility-client construction paths before
choosing a provider-native override.

## Gate Set

### Workstream Gate

```powershell
git diff --check -- docs/workstreams/native-extension-and-compat-retirement docs/workstreams/INDEX.md
```

### Registry Extension Factory Gate

```powershell
cargo check -p siumai-registry --tests --no-default-features --features openai
cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-default-features --features openai --no-fail-fast
```

### Method-Style / Generic-Client Gate

Use focused guards selected during NECR-040. At minimum they must prove remaining
`ProviderCompatibilityFactory` and `compat_*_client*` paths are compatibility-only and not stable
family execution.

### ContentPart Root-Move Gate

Use focused content projection and public import tests selected during NECR-050. Any namespace
movement must preserve serde-facing `ChatMessage` and `ChatResponse` compatibility.

## Evidence Log

| Date | Task | Command / Evidence | Result | What it proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | NECR-010 | Workstream docs opened. | Pass | Establishes the follow-on lane and task split for the three requested refactor areas. |
| 2026-05-25 | NECR-010 | `git diff --check -- docs/workstreams/native-extension-and-compat-retirement docs/workstreams/INDEX.md`. | Pass; Git reported the expected LF-to-CRLF working-copy warning for `docs/workstreams/INDEX.md`. | Proves the new workstream docs and index have no whitespace-error diff. |
| 2026-05-25 | NECR-020 | Source inventory with `rg -n "impl ProviderExtensionFactory\|compat_.*_client_with_ctx\|build_.*_extras_with_ctx\|as_.*_extras\|impl .*Extras" siumai-registry/src`. | Pass | Shows extension hooks are separate, while default image/speech/transcription extras still adapt `compat_*_client_with_ctx` unless providers override native methods. |
| 2026-05-25 | NECR-020 | Candidate review of `deepinfra.rs`, `fireworks.rs`, `togetherai.rs`, `openai.rs`, and `openai_compatible.rs`. | Pass | Selects DeepInfra, Fireworks, and TogetherAI image extras as the first safe native override set because they already expose native image clients implementing `ImageExtras`. |
| 2026-05-25 | NECR-030 | DeepInfra, Fireworks, and TogetherAI override `image_extras_with_ctx(...)` to return `Arc<dyn ImageExtras>` from their native image client builders. | Pass | Reduces extension-facet reliance on generic-client adapter fallback for three native-capable providers without changing public APIs. |
| 2026-05-25 | NECR-030 | `cargo fmt --package siumai-registry`. | Pass | Formats touched registry factory sources and boundary tests. |
| 2026-05-25 | NECR-030 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test hybrid_provider_image_extras_use_native_extension_clients --no-default-features --features openai,deepinfra,togetherai --no-fail-fast`. | Pass: 1 test run, 1 passed. | Guards that selected hybrid providers return native image extras clients and do not fall back through `compat_image_client_with_ctx`, `ClientBackedImageExtras`, or composite-client glue. |
| 2026-05-25 | NECR-030 | `cargo check -p siumai-registry --tests --no-default-features --features openai,deepinfra,togetherai`. | Pass | Proves the touched provider factories and tests compile under the feature set that covers the three native override targets. |
| 2026-05-25 | NECR-040 | Added `provider_compatibility_factory_is_method_style_only` and ADR-0007 retirement gates. | Pass | Converts generic-client retirement from prose into source-enforced guardrails and documented deletion prerequisites. |
| 2026-05-25 | NECR-040 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_compatibility_factory_is_method_style_only public_docs_classify_generic_llm_client_factory_paths_as_migration_only --no-default-features --features openai,deepinfra,togetherai --no-fail-fast`. | Pass: 2 tests run, 2 passed. | Proves `ProviderCompatibilityFactory` is production-confined to the facet definition plus historical `SiumaiBuilder` method-style construction and that public docs classify generic-client factory paths as migration-only. |
| 2026-05-25 | NECR-040 | `cargo check -p siumai-registry --tests --no-default-features --features openai,deepinfra,togetherai`. | Pass | Proves registry tests compile after adding the compatibility-factory guard and doc gates. |
| 2026-05-25 | NECR-050 | Added `adr_0008_root_content_part_move_has_serde_parity_fixture_gate`. | Pass | Adds a small executable serde fixture proving root and compat `ContentPart` paths currently serialize identically inside `ChatMessage` and `ChatResponse` payloads before any low-level root namespace move. |
| 2026-05-25 | NECR-050 | `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test adr_0008_root_content_part_move_has_serde_parity_fixture_gate adr_0008_full_contentpart_namespace_break_blockers_are_guarded --no-fail-fast`. | Pass: 2 tests run, 2 passed. | Proves ADR-0008 blockers remain documented and the new serde fixture gate reflects current root/compat payload parity. |
| 2026-05-25 | NECR-050 | `cargo check -p siumai-spec --tests --no-default-features`. | Pass | Proves spec tests compile after adding the ADR-0008 parity gate. |

## ADR-0008 Parity Notes

The first root-move fixture gate intentionally locks the current serde payload shape rather than
moving the namespace. It caught two important facts while being written:

- `MessageContent::MultiModal(...)` currently serializes as an externally tagged enum
  (`{"MultiModal":[...]}`), so root-move parity must preserve that shape unless a separate serde
  migration is approved.
- `ContentPart`-level metadata serializes as `providerMetadata`, while top-level `ChatResponse`
  metadata currently serializes as `provider_metadata`.

Those shapes are now executable evidence for future root namespace movement.

## Extension Factory Inventory

Default extension methods in `ProviderFactory` still adapt generic clients for six extension
surfaces:

- `file_management_capability_with_ctx(...)` via `compat_language_client_with_ctx(...)`
- `skills_capability_with_ctx(...)` via `compat_language_client_with_ctx(...)`
- `music_generation_capability_with_ctx(...)` via `compat_language_client_with_ctx(...)`
- `image_extras_with_ctx(...)` via `compat_image_client_with_ctx(...)`
- `speech_extras_with_ctx(...)` via `compat_speech_client_with_ctx(...)`
- `transcription_extras_with_ctx(...)` via `compat_transcription_client_with_ctx(...)`

The first native override set is intentionally limited to image extras because the selected
providers already have native image clients:

- DeepInfra: `DeepInfraImageClient` implements `ImageExtras`.
- Fireworks: `FireworksImageClient` implements `ImageExtras`.
- TogetherAI: `TogetherAiImageClient` implements `ImageExtras`.

Speech and transcription extras remain candidates for a separate provider-specific pass because the
current implementations often delegate through OpenAI-compatible text/audio clients rather than a
dedicated native extension object.

## Residual Risks

- Provider extension defaults may still adapt generic clients until each provider-native override is
  proven safe.
- Method-style construction and generic `LlmClient` remain available until ADR-0007 deletion
  prerequisites are satisfied.
- Low-level root `ContentPart` paths remain until ADR-0008 parity gates prove that the move is safe.
