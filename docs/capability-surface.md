# Capability Surface (Stable vs Extensions)

This document defines the *public contract* of Siumai during the fearless refactor.
The goal is to keep the stable surface small and predictable, while allowing provider-specific
features to evolve quickly.

## Stable Surface (Recommended)

The stable surface is intentionally limited to **six model families**:

1. Language (chat + streaming)
2. Embedding
3. Image generation
4. Reranking
5. Speech (TTS)
6. Transcription (STT)

These are exposed via `siumai::prelude::unified::*` and corresponding traits/handles.

### Design principles

- **Type-safe, provider-agnostic requests** for common parameters.
- **Streaming events are normalized** into a small number of events (`start`, `delta`, `usage`, `end` style).
- **Tools are first-class** (function tools + provider-executed tools).
- **No hard capability blocks**: capability checks are hints, not restrictions.

## Extensions Surface (Opt-in by Design)

Provider-specific features are exposed through **three explicit extension mechanisms**:

### 1) Provider-hosted tools

Provider-executed tools are modeled as `Tool::ProviderDefined` and constructed using helpers:

- `siumai::hosted_tools::openai::*`
- `siumai::hosted_tools::anthropic::*`
- `siumai::hosted_tools::google::*`

This keeps “web search”, “file search”, “code execution”, etc. out of the unified surface.

### 2) Provider options (pass-through)

Provider options are additional, provider-specific request inputs that should be:

- **Encapsulated within the provider**
- **Opaque to the core**
- **Backwards-compatible to evolve**

The Vercel-aligned shape is:

```text
providerOptions: Map<provider_id, JSON-object>
```

This allows shipping new provider features without editing core enums or shared structs.

### 3) Provider extension modules

Provider-specific endpoints/resources that do not fit the stable model families are exposed as:

- `siumai::provider_ext::<provider>::*`

Examples include provider-specific streaming formats, special endpoints, and non-family resources.

## Experimental (Unstable)

Low-level building blocks (executors, middleware, auth providers, etc.) are available under:

- `siumai::experimental::*`

This is an advanced surface for integrations and custom providers and is **not** part of the stable facade.

## What should *not* be in the stable surface

- Provider-specific protocol objects and raw API schemas
- Provider feature flags like “web search”, “prompt caching”, “thinking replay”
- Provider-specific streaming formats that cannot be normalized cleanly
