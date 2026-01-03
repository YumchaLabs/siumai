# Alpha.5 Architecture Snapshot (beta.5)

This document is a lightweight snapshot of the current refactor state in `0.11.0-beta.5`,
focused on **where protocol standards live** and **how provider-specific extensions are owned**.

## Where `standards/*` lives now

Provider-specific protocol mapping is provider-owned:

- OpenAI-like standard: `siumai-provider-openai-compatible/src/standards/openai/*`
- Anthropic Messages standard: `siumai-provider-anthropic-compatible/src/standards/anthropic/*`
- Gemini standard: `siumai-provider-gemini/src/standards/gemini/*`
- Ollama standard: `siumai-provider-ollama/src/standards/ollama/*`

Providers that reuse another provider’s standard may not have a `src/standards/*` directory.

Note: `siumai-core` may host **protocol-level shared building blocks** under `siumai-core/src/standards/*`
(e.g. OpenAI-compatible wire helpers). These are not provider-specific mapping modules.

## Provider-specific typed options / typed metadata ownership

Guiding rule (Vercel-aligned): `siumai-core` is provider-agnostic; **typed provider options and metadata are
owned by the provider crate** and surfaced via stable facade paths.

Current state:

- OpenAI typed options: `siumai-provider-openai/src/provider_options/openai/*`
- OpenAI typed metadata: `siumai-provider-openai/src/provider_metadata/openai.rs`
- Anthropic typed options: `siumai-provider-anthropic/src/provider_options/anthropic/*`
- Anthropic typed metadata: `siumai-provider-anthropic-compatible/src/provider_metadata/anthropic.rs` (re-exported by `siumai-provider-anthropic`)
- Gemini typed options: `siumai-provider-gemini/src/provider_options/gemini/*`
- Gemini typed metadata: `siumai-provider-gemini/src/provider_metadata/gemini.rs`
- Groq typed options: `siumai-provider-groq/src/provider_options/*`
- xAI typed options: `siumai-provider-xai/src/provider_options/*`
- Ollama typed options: `siumai-provider-ollama/src/provider_options/*`
- MiniMaxi typed options: `siumai-provider-minimaxi/src/provider_options/*`

Stable facade exports:

- `siumai::provider_ext::openai::*`
- `siumai::provider_ext::anthropic::*`
- `siumai::provider_ext::gemini::*`
- `siumai::provider_ext::groq::*`
- `siumai::provider_ext::xai::*`
- `siumai::provider_ext::ollama::*`
- `siumai::provider_ext::minimaxi::*`

## Provider options transport (Vercel-aligned)

Requests carry provider options as an **open, provider-id keyed JSON map**:

- `request.provider_options_map["<provider_id>"] = <json object>`

There is no longer a closed `ProviderOptions` enum transport in `siumai-core` (breaking change).
Typed option structs are provider-owned and exposed via `siumai::provider_ext::<provider>::*`.
