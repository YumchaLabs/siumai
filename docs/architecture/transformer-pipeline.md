# Siumai Transformer Pipeline (Chat Flow Overview)

This document describes how a chat request flows through the Siumai
transformer pipeline, from the public `ChatRequest` API down to
provider-specific HTTP calls and back to streamed events.

The goal is to make it easy for contributors to:

- Understand how core / standards / providers / aggregator interact.
- Add new providers (or standards) that plug into the same pipeline.
- Reason about where to put new logic when behaviour changes.

---

## High-level Data Flow

For chat, the pipeline is:

```text
User code
    ↓
ChatRequest (aggregator types)
    ↓
ProviderSpec (per provider, in `siumai`)
    ↓
CoreProviderContext + ChatInput (core types)
    ↓
Standard transformers (OpenAI / Anthropic / Gemini)
    ↓
HTTP execution (reqwest + middleware + retry)
    ↓
SSE stream (EventSource)
    ↓
ChatStreamEventCore (core streaming events)
    ↓
ChatStreamEvent (aggregator streaming events)
```

Each layer has a narrow responsibility and uses types from the layer
below it. This keeps cross-provider logic reusable while letting
providers customise behaviour where necessary.

---

## Layers and Responsibilities

### 1. Aggregator Types (`siumai` crate)

- **ChatRequest / ChatResponse / ChatStreamEvent**  
  Located in `siumai/src/types.rs` and `siumai/src/streaming.rs`.
  These are the user-facing types exposed by the `siumai` crate.

- **ProviderSpec / ProviderContext**  
  Located in `siumai/src/core/provider_spec.rs`.  
  Each provider (OpenAI, Anthropic, MiniMaxi, etc.) implements a
  `ProviderSpec` that:
  - Builds HTTP headers (`build_headers`).
  - Decides chat URL (`chat_url`).
  - Chooses transformers (`choose_chat_transformers`).
  - Optionally injects JSON hooks (`chat_before_send`).

The aggregator does not know about provider-specific JSON shapes. It
only works with:

- Aggregator-level request/response types.
- Abstract transformer traits.

### 2. Core Types (`siumai-core` crate)

- **ChatInput / ChatResult**  
  Located in `crates/siumai-core/src/execution/chat.rs`.  
  Minimal, provider-agnostic chat shapes used by standards:
  - `ChatInput`: messages + basic params (model, max_tokens, temperature, etc.).
  - `ChatResult`: plain text content + string `finish_reason` + usage metadata.

- **CoreProviderContext / CoreProviderSpec**  
  Located in `crates/siumai-core/src/provider_spec.rs`.  
  This is the core execution interface that standards and provider
  crates depend on. It does not depend on the `siumai` aggregator.

- **ChatStreamEventCore / ChatStreamEventConverterCore**  
  Located in `crates/siumai-core/src/execution/streaming.rs`.  
  Represents minimal streaming events that providers/standards can emit:
  - `ContentDelta`, `ToolCallDelta`, `ThinkingDelta`,
    `UsageUpdate`, `StreamStart`, `Custom`, `Error`.

### 3. Standards Crates (`siumai-std-*`)

Examples:

- `siumai-std-openai` → `OpenAiChatStandard`
- `siumai-std-anthropic` → `AnthropicChatStandard`
- `siumai-std-gemini` → `GeminiChatStandard`

Each standard crate:

- Depends only on `siumai-core`.
- Knows the provider's official JSON shape.
- Provides request/response/streaming transformers that operate on:
  - `ChatInput` / `ChatResult` / `ChatStreamEventCore`.

For example, `OpenAiChatStandard`:

- Converts `ChatInput` → OpenAI Chat Completions JSON body.
- Parses OpenAI Chat JSON → `ChatResult`.
- Converts SSE events (JSON strings) → `ChatStreamEventCore` values.

### 4. Provider Crates (`siumai-provider-*`)

Examples:

- `siumai-provider-openai` → `OpenAiCoreSpec`
- `siumai-provider-anthropic` → `AnthropicCoreSpec`
- `siumai-provider-minimaxi` → `MinimaxiCoreSpec`
- `siumai-provider-openai-compatible`

Each provider crate:

- Depends on `siumai-core` and the relevant `siumai-std-*` crates.
- Implements `CoreProviderSpec` using:
  - Standard transformers (e.g. `OpenAiChatStandard`).
  - Provider-specific header / routing helpers.

The aggregator `siumai` crate uses feature gates like
`provider-openai-external` / `provider-anthropic-external` to choose
between:

- In-crate, legacy implementations, or
- External provider crate implementations (`siumai-provider-*`),
  wired through `CoreProviderSpec`.

---

## Bridging Aggregator and Core

The key bridge lives in `siumai/src/core/provider_spec.rs`:

- `ProviderContext::to_core_context()`  
  Converts aggregator `ProviderContext` → `CoreProviderContext` for use
  by external provider crates.

- `bridge_core_chat_transformers`  
  Wraps `CoreChatTransformers` (request / response / stream) into
  aggregator `ChatTransformers` by:
  - Mapping `ChatRequest` → `ChatInput` via a function
    (`to_core_input`).
  - Mapping `ChatStreamEventCore` → `ChatStreamEvent` via another
    function (`map_stream_event`).

- `anthropic_like_chat_request_to_core_input`  
  Shared helper for Anthropic-style providers that maps
  `ChatRequest` → `ChatInput` with:
  - Minimal role mapping (System/User/Assistant).
  - Simple text content (`all_text()`).
  - Common params (model, max_tokens, temperature, etc.).

- `anthropic_like_map_core_stream_event`  
  Shared helper that maps `ChatStreamEventCore` into aggregator
  `ChatStreamEvent`, injecting a provider name into the
  `StreamStart` metadata. Used by Anthropic and MiniMaxi, and
  suitable for any other Anthropic-compatible provider.

Provider `Spec` implementations (in `siumai/src/providers/*/spec.rs`)
use these helpers to avoid duplicating mapping logic.

---

## Adding a New Provider (Chat Only)

When adding a provider that uses a known standard (e.g. Anthropic
Messages, OpenAI Chat, Gemini):

1. Implement a `CoreProviderSpec` in a new `siumai-provider-*` crate:
   - Declare capabilities.
   - Implement `build_headers`.
   - Implement `chat_url`.
   - Wire in the appropriate standard (`OpenAiChatStandard`,
     `AnthropicChatStandard`, etc.).

2. In the aggregator `siumai` crate:
   - Add a `ProviderSpec` implementation that:
     - Uses `ProviderContext::to_core_context()`.
     - Calls the external `CoreProviderSpec`.
     - Uses `bridge_core_chat_transformers` +
       a mapping helper (`anthropic_like_*` or a provider-specific one)
       to connect core transformers to aggregator types.

3. Optionally, add convenience builders and registry entries so that
   `LlmBuilder` and `ProviderRegistry` expose the new provider.

This pattern keeps provider-specific header/routing logic close to the
provider crate while reusing shared standards and core abstractions.

