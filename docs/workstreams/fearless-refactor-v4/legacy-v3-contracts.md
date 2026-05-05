# Legacy V3 Cleanup Boundary

This note defines which `V3` names are intentional AI SDK provider contracts and which names are
Siumai-local legacy surfaces that should not appear in the current public API.

## Intentional provider contracts

Keep `V3` or `V4` names when they mirror upstream contracts in `repo-ref/ai/packages/provider/src`:

- `LanguageModelV3` / `LanguageModelV4`
- `EmbeddingModelV3` / `EmbeddingModelV4`
- `ImageModelV3` / `ImageModelV4`
- `SpeechModelV3` / `SpeechModelV4`
- `TranscriptionModelV3` / `TranscriptionModelV4`
- `VideoModelV3` / `VideoModelV4`
- `RerankingModelV3` / `RerankingModelV4`

These names describe upstream provider protocol versions. They should only be used when the Rust
type is intentionally modeling that upstream provider contract.

## Removed local legacy names

Do not introduce or re-export Siumai-local `V3` names that are not upstream provider contracts:

- `TextModelV3`
- `CompletionModelV3`
- `LanguageModelV3StreamPart`
- `RerankModelV3`

Current Rust helpers should use the canonical family traits instead:

- `TextModel` / `LanguageModel`
- `CompletionModel`
- `RerankingModel`
- `TypedStreamPart` for the provider-agnostic typed stream overlay
- `LanguageModelV4StreamPart` only for the AI SDK V4 provider-facing stream union

`RerankCapability` remains available as a low-level non-unified extension under
`siumai::extensions`; it should not be imported by `prelude::unified::*`.

Historical workstream notes may still mention removed names as design history. Current examples,
tests, registry handles, and facade exports should not depend on them.

## Audit command

Use this as the first-pass cleanup check, then classify any remaining hits as either intentional
upstream contracts, model identifiers, or historical documentation:

```powershell
rg -n "\b(TextModelV3|CompletionModelV3|LanguageModelV3StreamPart|RerankModelV3)\b" siumai-core/src siumai/src siumai-registry/src siumai/tests docs CHANGELOG.md -g "*.rs" -g "*.md"
```
