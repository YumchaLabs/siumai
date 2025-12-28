# Alpha.5 Architecture Snapshot (beta.5)

This document is a lightweight snapshot of the current refactor state in `0.11.0-beta.5`,
focused on **where protocol standards live** and **how provider-specific extensions are owned**.

## Where `standards/*` lives now

`standards/*` is provider-owned (not in `siumai-core`):

- OpenAI-like standard: `siumai-provider-openai/src/standards/openai/*`
- Anthropic standard: `siumai-provider-anthropic/src/standards/anthropic/*`
- Gemini standard: `siumai-provider-gemini/src/standards/gemini/*`
- Ollama standard: `siumai-provider-ollama/src/standards/ollama/*`

Providers that reuse another provider’s standard may not have a `src/standards/*` directory.

## Provider-specific typed options / typed metadata ownership

Guiding rule (Vercel-aligned): `siumai-core` is provider-agnostic; **typed provider options and metadata are
owned by the provider crate** and surfaced via stable facade paths.

Current state:

- OpenAI typed options: `siumai-provider-openai/src/provider_options/openai/*`
- OpenAI typed metadata: `siumai-provider-openai/src/provider_metadata/openai.rs`
- Anthropic typed options: `siumai-provider-anthropic/src/provider_options/anthropic/*`
- Anthropic typed metadata: `siumai-provider-anthropic/src/provider_metadata/anthropic.rs`
- Gemini typed options: `siumai-provider-gemini/src/provider_options/gemini/*`
- Gemini typed metadata: `siumai-provider-gemini/src/provider_metadata/gemini.rs`

Stable facade exports:

- `siumai::provider_ext::openai::*`
- `siumai::provider_ext::anthropic::*`

## Provider options transport (compatibility layer)

Requests carry provider options in two forms during the refactor:

1. Preferred: an open, provider-id keyed JSON map: `request.provider_options_map["<provider_id>"] = <json>`.
2. Compatibility: `ProviderOptions::<Provider>(serde_json::Value)` (JSON payload, not typed structs).

Providers should parse options from the map first, and fall back to the enum variant if present.
