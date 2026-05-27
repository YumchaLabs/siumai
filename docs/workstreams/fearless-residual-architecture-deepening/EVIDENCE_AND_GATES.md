# Fearless Residual Architecture Deepening - Evidence And Gates

Status: Closed
Last updated: 2026-05-27

## Baseline Evidence

| Date | Task | Command / Evidence | Result | Notes |
| --- | --- | --- | --- | --- |
| 2026-05-27 | FRAD-010 | Architecture review using `improve-codebase-architecture`; report opened from the OS temp directory. | Pass | Found five candidates: registry descriptor, OpenAI-compatible message dialects, bridge codecs, provider contract harness, and ADR-0008 `ContentPart` root move. |
| 2026-05-27 | FRAD-010 | `python -m json.tool docs\workstreams\fearless-residual-architecture-deepening\WORKSTREAM.json` | Pass | Workstream metadata parses and records FRAD-020 as the first executable task. |
| 2026-05-27 | FRAD-010 | `git diff --check -- docs\workstreams\fearless-residual-architecture-deepening docs\workstreams\INDEX.md` | Pass | Diff check reported only expected LF-to-CRLF working-copy warnings for `INDEX.md`. |
| 2026-05-27 | FRAD-020 | `cargo fmt --check -p siumai-registry` | Pass | Registry formatting is clean after extracting provider descriptor seams. |
| 2026-05-27 | FRAD-020 | `cargo check -p siumai-registry --features openai,azure,anthropic,google,google-vertex,ollama,xai,groq,deepseek,deepinfra,minimaxi,cohere,togetherai,bedrock,gateway,deepgram,elevenlabs` | Pass | Full built-in provider feature set compiles through the descriptor split. |
| 2026-05-27 | FRAD-020 | `cargo check -p siumai-registry --no-default-features` | Pass | Feature-minimal registry still compiles; existing unused compatibility warnings remain unrelated. |
| 2026-05-27 | FRAD-020 | `cargo nextest run -p siumai-registry provider_catalog --no-fail-fast` | Pass | 1 provider catalog boundary test passed. |
| 2026-05-27 | FRAD-020 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-fail-fast` | Pass | 36 registry architecture boundary tests passed, including the new provider descriptor source guard. |
| 2026-05-27 | FRAD-030 | `cargo fmt --check -p siumai-protocol-openai` | Pass | Protocol formatting is clean after extracting message dialect conversion and tests. |
| 2026-05-27 | FRAD-030 | `cargo check -p siumai-protocol-openai --features openai-standard,openai-responses` | Pass | OpenAI protocol crate compiles with Chat Completions and Responses surfaces enabled. |
| 2026-05-27 | FRAD-030 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses standards::openai::utils::message_dialect --no-fail-fast` | Pass | 23 dialect-local tests passed for OpenAI Chat, OpenAI-compatible, Perplexity, DeepSeek, xAI, and Mistral message conversion. |
| 2026-05-27 | FRAD-030 | `cargo nextest run -p siumai-protocol-openai --features openai-standard,openai-responses openai --no-fail-fast` | Pass | 477 OpenAI protocol tests passed, with 2 expected skips. |
| 2026-05-27 | FRAD-040 | `cargo check -p siumai-bridge --features openai,anthropic,google` | Pass | Bridge crate compiles with all request codec families enabled. |
| 2026-05-27 | FRAD-040 | `cargo fmt --check -p siumai-bridge` | Pass | Bridge formatting is clean after splitting request codecs. |
| 2026-05-27 | FRAD-040 | `cargo nextest run -p siumai-bridge --features openai,anthropic,google request --no-fail-fast` | Pass | 49 request bridge tests passed, including the new per-wire-format codec source guard; 63 skipped by filter. |
| 2026-05-27 | FRAD-050 | `cargo fmt --check -p siumai-registry -p siumai` | Pass | Registry and facade formatting is clean after deepening provider contract/public-path harnesses. |
| 2026-05-27 | FRAD-050 | `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test --no-fail-fast` | Pass | 37 registry architecture boundary tests passed, including named family override scenario and provider public-path harness guards; existing unused compatibility warnings remain unrelated. |
| 2026-05-27 | FRAD-050 | `cargo nextest run -p siumai --test provider_public_path_parity_test --features openai,azure,anthropic,google,google-vertex,xai,groq,cohere,togetherai,deepinfra,bedrock,deepseek,ollama,minimaxi --no-fail-fast` | Pass | 506 facade public-path parity tests passed across built-in provider families. |
| 2026-05-27 | FRAD-060 | `cargo fmt --check -p siumai-spec -p siumai-core -p siumai` | Pass | Spec/core/facade formatting is clean after moving production legacy `ContentPart` paths to explicit compat imports. |
| 2026-05-27 | FRAD-060 | `cargo check -p siumai-core -p siumai-spec -p siumai` | Pass | Spec/core/facade crates compile with default provider features after compat import migration. |
| 2026-05-27 | FRAD-060 | `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test adr_0008 --no-fail-fast` | Pass | 3 ADR-0008 focused tests passed, including serde parity, documented root-move blockers, and explicit production compat import guards. |
| 2026-05-27 | FRAD-060 | `cargo nextest run -p siumai-spec content --no-fail-fast` | Pass | 39 content/projection tests passed, including the new ADR-0008 production compat import guard. |
| 2026-05-27 | FRAD-060 | `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast` | Pass | 23 public-surface import tests passed; legacy `ContentPart` remains outside the stable unified prelude and explicit compat imports compile. |
| 2026-05-27 | FRAD-060 | `cargo nextest run -p siumai --test facade_architecture_boundary_test content --no-fail-fast` | Pass | 6 content facade architecture tests passed after updating content-part audit records for FRAD-030/040 codec/dialect splits and FRAD-060 compat imports. |
| 2026-05-27 | FRAD-060 | `cargo nextest run -p siumai test_macros --no-fail-fast` | Pass | Facade macro expansion still compiles while `tool!` now uses the private compat content path. |
| 2026-05-27 | FRAD-070 | `review-workstream` closeout review | Pass | No blocking workstream or code-quality findings remained after fixing closeout-gate findings. Residual ADR-0008 root move blocker remains recorded as an intentional follow-on condition, not open work in this lane. |
| 2026-05-27 | FRAD-070 | `cargo nextest run -p siumai-registry -p siumai-protocol-openai -p siumai-bridge -p siumai-spec -p siumai-core -p siumai --no-fail-fast` | Initial fail | 1995 passed, 4 failed. The failed gates exposed OpenAI Responses hosted tool-result `providerExecuted` loss, a stale video facade guard, and a feature-gated Vertex xAI audio boundary guard. These were fixed before closeout. |
| 2026-05-27 | FRAD-070 | `cargo fmt --check -p siumai-registry -p siumai-protocol-openai -p siumai-bridge -p siumai-spec -p siumai-core -p siumai` | Pass | Six touched crates are formatting-clean after closeout fixes. |
| 2026-05-27 | FRAD-070 | `cargo nextest run -p siumai --test facade_architecture_boundary_test facade_video_metadata_projection_avoids_legacy_request_provider_options --no-fail-fast` | Pass | Stale facade video source guard now follows the workflow helper seam without cutting production source at an early test-only import. |
| 2026-05-27 | FRAD-070 | `cargo nextest run -p siumai --test openai_compatible_audio_boundary_test compat_registry_audio_handles_follow_declared_capability_split --no-fail-fast` | Pass | Feature-gated Vertex MaaS / Vertex xAI registry paths are no longer misclassified as default OpenAI-compatible audio handles. |
| 2026-05-27 | FRAD-070 | `cargo nextest run -p siumai --test openai_responses_response_fixtures_alignment_test openai_responses_response_fixtures_match --no-fail-fast` | Pass | OpenAI Responses hosted MCP/code/image/file/web tool-result fixtures keep provider-executed ownership in compatibility `ContentPart` projection. |
| 2026-05-27 | FRAD-070 | `cargo nextest run -p siumai --test openai_responses_response_bridge_roundtrip_fixtures_alignment_test openai_responses_response_bridge_roundtrip_fixture_exact_cases_match --no-fail-fast` | Pass | Bridge roundtrips preserve provider-executed hosted tool results for exact OpenAI Responses fixtures. |
| 2026-05-27 | FRAD-070 | `cargo nextest run -p siumai-registry -p siumai-protocol-openai -p siumai-bridge -p siumai-spec -p siumai-core -p siumai --no-fail-fast` | Pass | Final closeout gate: 1999 tests passed, 6 skipped. |
| 2026-05-27 | FRAD-070 | `python -m json.tool docs\workstreams\fearless-residual-architecture-deepening\WORKSTREAM.json` | Pass | Closed workstream metadata parses. |
| 2026-05-27 | FRAD-070 | `git diff --check -- docs\workstreams\fearless-residual-architecture-deepening docs\workstreams\INDEX.md` | Pass | Closeout docs diff check is clean aside from expected LF-to-CRLF working-copy warnings. |

## Required Gates

### Planning Gate

```text
python -m json.tool docs/workstreams/fearless-residual-architecture-deepening/WORKSTREAM.json
git diff --check -- docs/workstreams/fearless-residual-architecture-deepening docs/workstreams/INDEX.md
```

### Slice Gates

- Use crate-scoped `cargo fmt --check -p <crate>` for touched crates.
- Use targeted `cargo nextest run -p <crate> <filter> --no-fail-fast` during iteration.
- Run package gates before marking a slice done when the slice changes shared behavior.
- Record any Windows path-length error 206 if a workspace-wide command fails for environmental
  reasons.

### Closeout Gate

```text
cargo fmt --check -p siumai-registry -p siumai-protocol-openai -p siumai-bridge -p siumai-spec -p siumai-core -p siumai
cargo nextest run -p siumai-registry -p siumai-protocol-openai -p siumai-bridge -p siumai-spec -p siumai-core -p siumai --no-fail-fast
```

## Review Gate

Run `review-workstream` before accepting major slices and before closeout. Review must check:

- ADR-0001/0002 provider-first ownership alignment.
- ADR-0006/0007 family-model-first and `LlmClient` demotion alignment.
- ADR-0008 gates before any root `ContentPart` public break.
- ADR-0009 consistency for AI SDK-facing contract seams.
- Changelog coverage for behavior-visible or public contract changes.
