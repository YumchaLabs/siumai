# Fearless Residual Architecture Deepening - Evidence And Gates

Status: Active
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
