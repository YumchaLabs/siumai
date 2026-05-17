# AI SDK Provider Interface Convergence - Evidence And Gates

Status: Active
Last updated: 2026-05-18

## Smallest Current Repro

The active implementation slice is OpenAI-compatible promoted vendor package parity after the core,
registry, protocol stream, and bridge/gateway guard slices completed.

```bash
cargo nextest run -p siumai-core --no-fail-fast
cargo nextest run -p siumai-registry --no-fail-fast
cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
```

## Gate Set

### Program Document Gate

```bash
git status --short
```

Proves the program docs and index changes are the only changes in the opening commit.

### Core Seam Gate

```bash
cargo nextest run -p siumai-core --no-fail-fast
```

Proves provider-agnostic runtime guards and core tests still pass after seam changes.

### Registry Seam Gate

```bash
cargo nextest run -p siumai-registry --no-fail-fast
```

Proves registry family handles, factories, caches, aliases, and source guards still pass after
family-first construction changes.

### Stream Convergence Gate

```bash
cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast
cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast
cargo nextest run -p siumai-bridge --no-fail-fast
```

Proves protocol-owned stream/parser/serializer changes preserve no-network fixture and bridge
behavior.

### Provider Parity Gate

```bash
cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast
cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast
```

Proves public package exports and promoted OpenAI-compatible vendor behavior do not drift while
package parity slices land.

### Broad Closeout Gate

```bash
cargo nextest run --profile ci --all-features --workspace
```

Proves the full workspace still passes after this program closes. On local Windows, a package-scoped
format check may be used when `cargo fmt --all -- --check` hits command-line length limits; CI
should still run the full formatting gate.

## Evidence Anchors

- `docs/workstreams/ai-sdk-provider-interface-convergence/DESIGN.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/TODO.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/MILESTONES.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/PARITY_INVENTORY.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/HANDOFF.md`
- `docs/workstreams/ai-sdk-provider-interface-convergence/WORKSTREAM.json`
- `docs/adr/0001-vercel-aligned-modular-split.md`
- `docs/adr/0006-family-model-first-trait-policy.md`
- `docs/adr/0007-llmclient-demotion-policy.md`
- `docs/adr/0008-legacy-content-part-compatibility-boundary.md`

## Command Log

| Date | Command | Result | Notes |
| --- | --- | --- | --- |
| 2026-05-18 | `git status --short --branch` | Passed | Opening worktree was clean on `main`. |
| 2026-05-18 | `cargo fmt -p siumai-core --check` | Passed | Formatting gate for the AIPC-030 core guard slice. |
| 2026-05-18 | `cargo nextest run -p siumai-core core_standards_module_does_not_reintroduce_provider_protocol_islands --no-fail-fast` | Passed | Proves `siumai-core::standards` cannot regain provider/protocol island directories or modules. |
| 2026-05-18 | `cargo nextest run -p siumai-registry stable_registry_handles_do_not_use_compat_client_paths_for_primary_family_execution --no-fail-fast` | Passed | Reconfirmed the existing registry stable-family handle guard. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --no-fail-fast` | Passed | 96 tests passed; existing warnings are unrelated dead-code warnings in test support. |
| 2026-05-18 | `cargo nextest run -p siumai-core --no-fail-fast` | Passed | 426 tests passed after moving the new guard into the existing integration boundary test suite. |
| 2026-05-18 | `cargo fmt -p siumai-registry --check` | Passed | Formatting gate for the AIPC-040 registry handle boundary slice. |
| 2026-05-18 | `cargo nextest run -p siumai-registry remaining_registry_handle_compat_paths_are_extension_only --no-fail-fast` | Passed | Proves remaining registry compat client paths are isolated to extension-only image/audio helpers. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --no-fail-fast` | Passed | 97 tests passed; existing warnings are unrelated dead-code warnings in test support. |
| 2026-05-18 | `cargo fmt -p siumai-protocol-openai --check` | Passed | Formatting gate for the OpenAI Responses AIPC-050 stream-part slice. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses --test responses_sse_feature_surface_test --no-fail-fast` | Passed | Proves the public OpenAI Responses feature surface roundtrips stable `ChatStreamEvent::Part` tool call/result parts. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-openai responses_feature_surface_uses_stable_parts_for_tool_stream_parts --no-fail-fast` | Passed | Proves the public OpenAI Responses feature surface does not regress to provider custom events for stable tool stream parts. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast` | Passed | 448 tests passed for the OpenAI protocol package after the first AIPC-050 slice. |
| 2026-05-18 | `cargo fmt -p siumai-extras --check` | Passed | Formatting gate for the AIPC-050 gateway stable-part assertion slice. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai,anthropic gateway_smoke_asserts_stable_tool_stream_parts --no-fail-fast` | Passed | Proves gateway smoke tests no longer accept provider custom payloads for stable tool stream part assertions. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai,anthropic gateway_route_smoke_transcodes_anthropic_fixture_to_openai_sse --no-fail-fast` | Passed | Proves the Anthropic-to-OpenAI Responses gateway still emits stable tool call/result parts after the helper was tightened. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai,anthropic,google --test gateway_axum_smoke_test --no-fail-fast` | Passed | 20 gateway Axum smoke tests passed; existing Gemini unreachable-pattern warning is unrelated to this slice. |
| 2026-05-18 | `cargo fmt -p siumai-protocol-anthropic -p siumai-protocol-gemini --check` | Passed | Formatting gate for the Anthropic/Gemini serializer custom-input boundary slice. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-anthropic --all-features anthropic_streaming_serializer_custom_inputs_are_compat_or_provider_native_only --no-fail-fast` | Passed | Proves Anthropic serializer tests only use custom-event inputs for explicit V3 compatibility or provider-native custom cases. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-gemini --all-features gemini_streaming_serializer_custom_inputs_are_compat_or_provider_native_only --no-fail-fast` | Passed | Proves Gemini serializer tests only use custom-event inputs for explicit V3 compatibility or provider-native custom cases. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-anthropic --all-features --no-fail-fast` | Passed | 219 Anthropic protocol tests passed after the serializer boundary slice. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-gemini --all-features --no-fail-fast` | Passed | 138 Gemini protocol tests passed after the serializer boundary slice. |
| 2026-05-18 | `cargo fmt -p siumai-bridge -p siumai-extras -- --check` | Passed | Formatting gate for the first AIPC-060 bridge/gateway slice. |
| 2026-05-18 | `cargo nextest run -p siumai-bridge --features openai,google openai_responses_stream_bridge_maps_stable_provider_tool_parts_to_output_items --no-fail-fast` | Passed | Proves OpenAI Responses stream bridging emits output item frames from stable provider-tool stream parts, not only custom compatibility events. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai openai_responses_gateway_parts_adapter_uses_bridge_stream_seam_directly --no-fail-fast` | Passed | Proves extras gateway code imports the OpenAI Responses stream parts adapter through `siumai_bridge::stream`. |
| 2026-05-18 | `cargo nextest run -p siumai-bridge --all-features --no-fail-fast` | Passed | 108 bridge tests passed after the stable provider-tool stream part bridge regression test. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai,anthropic,google --test gateway_axum_smoke_test --no-fail-fast` | Passed | 20 gateway Axum smoke tests passed after the extras seam import cleanup; existing Gemini unreachable-pattern warning is unrelated. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai --test bridge_architecture_boundary_test --no-fail-fast` | Passed | 3 extras bridge boundary tests passed, including the new direct `siumai_bridge::stream` adapter guard. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai openai_responses_direct_helper_bridges_stable_provider_tool_parts --no-fail-fast` | Passed | Proves the direct extras OpenAI Responses SSE helper bridges stable provider-tool stream parts into output item frames. |
| 2026-05-18 | `cargo fmt -p siumai-extras -- --check` | Passed | Formatting gate after the direct extras SSE helper regression test. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai --lib transcode_tests --no-fail-fast` | Passed | 18 extras Axum transcode tests passed, including the direct stable provider-tool stream part helper regression. |
| 2026-05-18 | `cargo nextest run -p siumai-extras --features server,openai,anthropic,google --test gateway_axum_smoke_test --no-fail-fast` | Passed | 20 gateway Axum smoke tests passed after closing AIPC-060; existing Gemini unreachable-pattern warning is unrelated. |
| 2026-05-18 | `cargo fmt -p siumai-registry -p siumai-protocol-openai -p siumai-provider-openai-compatible -- --check` | Passed | Formatting gate for AIPC-070 explicit OpenAI-compatible completion capability metadata. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-openai-compatible --all-features promoted_vendor_clients_only_expose_explicit_completion_capability --no-fail-fast` | Passed | Proves promoted vendor clients expose completion only when the preset explicitly declares completion. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-openai-compatible --all-features test_provider_capabilities test_non_completion_promoted_presets_keep_completion_metadata_absent test_completion_capable_hybrid_presets_keep_explicit_completion_metadata --no-fail-fast` | Passed | Proves provider metadata matches the explicit completion matrix for promoted vendors and hybrid providers. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --all-features openai_compatible_factory_promoted_chat_vendors_do_not_inherit_completion_capability openai_compatible_factory_mistral_rejects_native_completion_family_path openai_compatible_factory_perplexity_rejects_native_completion_family_path vertex_maas_factory_supports_completion_and_embedding_family_paths openai_compatible_factory_together_supports_native_completion_family_path openai_compatible_factory_fireworks_supports_native_completion_family_path --no-fail-fast` | Passed | Proves registry factory completion family paths reject non-completion promoted vendors while keeping explicit completion providers and Vertex MaaS working. |
| 2026-05-18 | `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast` | Passed | 449 OpenAI protocol tests passed after changing completion surface inference to explicit metadata. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-openai-compatible --all-features --no-fail-fast` | Passed | 235 OpenAI-compatible provider tests passed after updating completion fixtures to declare explicit completion capability. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --all-features --no-fail-fast` | Passed | 523 registry tests passed after adding the promoted vendor completion inheritance contract. |
| 2026-05-18 | `cargo fmt -p siumai-registry -p siumai-provider-groq -- --check` | Passed | Formatting gate for the AIPC-080 Groq package boundary slice. |
| 2026-05-18 | `cargo nextest run -p siumai-provider-groq --all-features provider_owned_tts_models_stay_outside_chat_catalog --no-fail-fast` | Passed | 1 Groq provider catalog test passed, proving provider-owned TTS stays outside the AI SDK-aligned chat catalog. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --features groq provider_catalog_uses_native_metadata_for_groq groq_factory_declares_audio_without_image_or_rerank groq_ai_sdk_aligned_catalog_excludes_provider_owned_speech_models --no-fail-fast` | Passed | 3 Groq registry tests passed, locking the AI SDK chat/transcription surface versus provider-owned Rust speech extension boundary. |
| 2026-05-18 | `cargo fmt -p siumai-registry -- --check` | Passed | Formatting gate for the AIPC-080 TogetherAI package/audio-extension terminology slice. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --features togetherai provider_catalog_uses_native_metadata_for_togetherai togetherai_factory_exposes_unified_provider_capabilities togetherai_package_surface_and_audio_extension_boundary_is_explicit --no-fail-fast` | Passed | 3 TogetherAI registry tests passed, proving the capability surface remains intact while docs/tests separate AI SDK package families from Siumai audio extension families. |
| 2026-05-18 | `cargo fmt -p siumai-registry -- --check` | Passed | Formatting gate for the AIPC-080 xAI files/speech-boundary registry slice. |
| 2026-05-18 | `cargo nextest run -p siumai-registry --features xai provider_catalog_uses_native_metadata_for_xai xai_factory_returns_provider_owned_client xai_factory_declares_files_and_speech_boundary --no-fail-fast` | Passed | 3 xAI registry tests passed, proving registry metadata declares AI SDK-aligned files while speech remains a documented provider-owned Rust extension and completion/embedding/rerank/transcription stay unsupported. |
