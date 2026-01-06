# Alpha.5 Architecture Snapshot (beta.5)

This document is a lightweight snapshot of the current refactor state in `0.11.0-beta.5`,
focused on **where protocol standards live** and **how provider-specific extensions are owned**.

## Where `standards/*` lives now

Provider-specific protocol mapping is **protocol-crate-owned** (provider crates re-export for compatibility):

- OpenAI-like standard: `siumai-protocol-openai` (current impl: `siumai-provider-openai-compatible/src/standards/openai/*`)
- Anthropic Messages standard: `siumai-protocol-anthropic` (current impl: `siumai-provider-anthropic-compatible/src/standards/anthropic/*`)
- Gemini standard: `siumai-protocol-gemini/src/standards/gemini/*`
- Vertex Imagen standard: `siumai-provider-google-vertex/src/standards/vertex_imagen.rs`
- Ollama standard: `siumai-provider-ollama/src/standards/ollama/*`

Providers that reuse another provider’s standard may not have a `src/standards/*` directory.

Note: `siumai-core` may host **protocol-level shared building blocks** under `siumai-core/src/standards/*`
(e.g. OpenAI-compatible wire helpers). These are not provider-specific mapping modules.

## Provider-specific typed options / typed metadata ownership

Guiding rule (Vercel-aligned): `siumai-core` is provider-agnostic.

- Typed provider options are provider-owned.
- Typed response metadata may be provider-owned or protocol-family-owned (when shared across multiple providers).
  In either case, they are surfaced via stable facade paths.

Current state:

- OpenAI typed options: `siumai-provider-openai/src/provider_options/openai/*`
- OpenAI typed metadata: `siumai-protocol-openai` (current impl: `siumai-provider-openai-compatible/src/provider_metadata/openai.rs`; re-exported by `siumai-provider-openai`)
- Anthropic typed options: `siumai-provider-anthropic/src/provider_options/anthropic/*`
- Anthropic typed metadata: `siumai-protocol-anthropic` (current impl: `siumai-provider-anthropic-compatible/src/provider_metadata/anthropic.rs`; re-exported by `siumai-provider-anthropic`)
- Gemini typed options: `siumai-provider-gemini/src/provider_options/gemini/*`
- Gemini typed metadata: `siumai-provider-gemini/src/provider_metadata/gemini.rs`
- Vertex typed options: `siumai-provider-google-vertex/src/provider_options/vertex/*`
- Groq typed options: `siumai-provider-groq/src/provider_options/*`
- xAI typed options: `siumai-provider-xai/src/provider_options/*`
- Ollama typed options: `siumai-provider-ollama/src/provider_options/*`
- MiniMaxi typed options: `siumai-provider-minimaxi/src/provider_options/*`

Stable facade exports (stable module roots):

- `siumai::provider_ext::openai::*`
- `siumai::provider_ext::anthropic::*`
- `siumai::provider_ext::gemini::*`
- `siumai::provider_ext::groq::*`
- `siumai::provider_ext::xai::*`
- `siumai::provider_ext::ollama::*`
- `siumai::provider_ext::minimaxi::*`

For new code, prefer structured imports (to avoid accidental coupling):

- Typed options: `siumai::provider_ext::<provider>::options::*`
- Typed metadata: `siumai::provider_ext::<provider>::metadata::*` (when available)
- Escape hatches: `siumai::provider_ext::<provider>::ext::*`
- Extra resources: `siumai::provider_ext::<provider>::resources::*` (when available)

## Provider options transport (Vercel-aligned)

Requests carry provider options as an **open, provider-id keyed JSON map**:

- `request.provider_options_map["<provider_id>"] = <json object>`

There is no longer a closed `ProviderOptions` enum transport in `siumai-core` (breaking change).
Typed option structs are provider-owned and exposed via `siumai::provider_ext::<provider>::*`.
