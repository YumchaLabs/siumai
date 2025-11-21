# Provider Standards Pattern (OpenAI / Anthropic / Gemini)

This document explains how Siumai structures provider integrations
around **standards** (OpenAI / Anthropic / Gemini) plus thin
provider-specific glue. The goal is to make providers:

- Easy to reason about.
- Easy to extend (new params, new endpoints).
- Easy to keep in sync with Vercel AI SDK–style patterns.

It also serves as a template for adding new providers or capabilities
such as rerank.

---

## Design Goals

- **Standards first**  
  Put protocol–specific JSON (OpenAI, Anthropic, Gemini) in dedicated
  `siumai-std-*` crates that only depend on `siumai-core`.

- **Providers as adapters**  
  `siumai-provider-*` crates own provider-specific behaviour:
  headers, base URLs, routing quirks. They are thin on logic.

- **Aggregator as glue**  
  The top-level `siumai` crate wires standards and providers together
  using `ProviderSpec`, without knowing any provider JSON details.

This matches Vercel AI SDK’s architecture:

- core → standard adapters → provider adapters → runtime.

---

## The Three Layers

### 1. Standards (`siumai-std-*`)

Examples:

- `siumai-std-openai` → `OpenAiChatStandard`, `OpenAiEmbeddingStandard`
- `siumai-std-anthropic` → `AnthropicChatStandard`
- `siumai-std-gemini` → `GeminiChatStandard`, `GeminiEmbeddingStandard`, `GeminiImageStandard`

Responsibilities:

- Define **request/response/streaming transformers** based on
  `siumai-core` types, e.g.:
  - `ChatInput` / `ChatResult`
  - `ChatStreamEventCore`
  - `EmbeddingInput` / `EmbeddingResult`
  - Image request/response core types
- Implement the official JSON shapes:
  - Headers and base URLs live in provider crates, not here.
  - Standards are reusable across providers that share a protocol
    (e.g. OpenAI-compatible providers).


### 2. Providers (`siumai-provider-*`)

Examples:

- `siumai-provider-openai` → `OpenAiCoreSpec`
- `siumai-provider-anthropic` → `AnthropicCoreSpec`
- `siumai-provider-minimaxi` → `MinimaxiCoreSpec`
- `siumai-provider-openai-compatible`
- `siumai-provider-groq`, `siumai-provider-xai`

Responsibilities:

- Implement `CoreProviderSpec` using standards:
  - Select URLs based on standard + provider rules.
  - Build headers (API keys, org/project, custom headers).
  - Choose standard transformers for chat / embedding / image.
- Provide small helper functions:
  - `build_openai_json_headers` (OpenAI).
  - `build_anthropic_json_headers` (Anthropic).
  - `build_groq_json_headers` / `build_xai_json_headers`, etc.
- Own provider-specific policies (e.g. OpenRouter Referer rules).


### 3. Aggregator (`siumai` crate)

Responsibilities:

- Export user-facing types and traits:
  - `ChatRequest`, `ChatResponse`, `ChatStreamEvent`
  - `EmbeddingRequest`, `EmbeddingResponse`
  - `ImageGenerationRequest`, `ImageGenerationResponse`
- Implement `ProviderSpec` per provider:
  - `build_headers`, `chat_url`, `embedding_url`, `image_url`
  - `choose_chat_transformers`, `choose_embedding_transformers`,
    `choose_image_transformers`
  - `chat_before_send`, `embedding_before_send`, `image_before_send`
- Bridge between aggregator types and core types:
  - Use helpers in `core/provider_spec.rs` such as
    `openai_chat_request_to_core_input`,
    `anthropic_like_chat_request_to_core_input`,
    `gemini_like_chat_request_to_core_input`.
  - Use `bridge_core_chat_transformers` +
    `map_core_stream_event_with_provider` for streaming.

The aggregator should not need to know provider JSON details; it only
coordinates types and glue.

---

## Concrete Examples

### OpenAI (Chat / Embedding / Image)

- **Standard**: `siumai-std-openai`
  - `OpenAiChatStandard`, `OpenAiEmbeddingStandard`, `OpenAiImageStandard`.
  - Implement Chat / Embedding / Image JSON based on `ChatInput`,
    `EmbeddingInput`, and image core types.

- **Provider crate**: `siumai-provider-openai`
  - `OpenAiCoreSpec` builds URLs and headers, selects which standard
    to use.
  - `build_openai_json_headers` is the central headers helper.

- **Aggregator**:
  - `OpenAiSpec` in `siumai/src/providers/openai/spec.rs` implements
    `ProviderSpec`:
    - Uses `openai_chat_request_to_core_input` to map
      `ChatRequest` → `ChatInput`.
    - Chooses standard transformers for chat/embedding/image.
    - Delegates headers/URLs to `siumai-provider-openai` when feature
      `provider-openai-external` is enabled.
  - `OpenAiChatCapability` and `OpenAiClient` use `ChatExecutorBuilder`
    and `HttpChatExecutor` with the chosen transformers.


### Anthropic (Messages API)

- **Standard**: `siumai-std-anthropic`
  - `AnthropicChatStandard` implements Messages API based on
    `ChatInput` / `ChatResult` / `ChatStreamEventCore`.
  - Provides `AnthropicParsedContentCore` and helpers to capture:
    - aggregated text
    - tool calls
    - thinking blocks

- **Provider crate**: `siumai-provider-anthropic`
  - `AnthropicCoreSpec` wires Anthropic endpoints and headers into the
    core execution model.

- **Aggregator**:
  - `AnthropicSpec` in `siumai/src/providers/anthropic/spec.rs`:
    - Uses `anthropic_like_chat_request_to_core_input` for
      `ChatRequest` → `ChatInput`.
    - Bridges core transformers with `bridge_core_chat_transformers`.
    - Uses `anthropic_like_map_core_stream_event` to map
      `ChatStreamEventCore` → `ChatStreamEvent`.
    - Delegates headers to `siumai-provider-anthropic` when
      `provider-anthropic-external` is enabled.
  - `providers/anthropic/utils.rs` contains
    `core_parsed_to_message_content` to convert
    `AnthropicParsedContentCore` into aggregator `MessageContent`.


### Gemini (Chat / Embedding / Image)

- **Standard**: `siumai-std-gemini`
  - `GeminiChatStandard` + `GeminiEmbeddingStandard` +
    `GeminiImageStandard`.
  - `GeminiParsedContentCore` and `parse_content_core` unify:
    - text, tool calls, thinking content.

- **Provider crate**: `siumai-provider-gemini`
  - `GeminiCoreSpec` implements `CoreProviderSpec` for chat using
    the Gemini Chat standard.
  - `build_gemini_json_headers` centralises Gemini API header
    construction (API key vs. Authorization/Bearer).

- **Aggregator**:
  - `GeminiSpec` in `siumai/src/providers/gemini/spec.rs`:
    - Uses `gemini_like_chat_request_to_core_input` to map
      `ChatRequest` → `ChatInput` and propagate `GeminiOptions` into
      `ChatInput::extra`.
    - Uses `GeminiChatStandard` (via `std-gemini-external`) for chat,
      bridged with `bridge_core_chat_transformers`.
    - Uses `GeminiEmbeddingStandard` / `GeminiImageStandard` for
      embedding/image transformers (`choose_embedding_transformers`,
      `choose_image_transformers`), always via `std-gemini-external`.
    - Delegates Gemini headers to `siumai-provider-gemini` when
      `provider-gemini-external` is enabled; otherwise falls back to
      the in-crate implementation.
  - `GeminiChatCapability` and `GeminiClient` call:
    - `GeminiSpec::choose_chat_transformers` +
      `HttpChatExecutor` for chat/stream.
    - `GeminiSpec::choose_embedding_transformers` +
      `HttpEmbeddingExecutor` for embedding.
    - `GeminiSpec::choose_image_transformers` +
      `HttpImageExecutor` for image generation.
  - `providers/gemini/utils.rs` provides
    `core_parsed_to_content_parts` to convert `GeminiParsedContentCore`
    into `ContentPart` + aggregated text.

This makes Gemini follow the same “standard + provider + aggregator
bridge” pattern as OpenAI and Anthropic. Legacy, in-crate Gemini
standards are no longer used when `std-gemini-external` is enabled;
without this feature, Gemini capabilities report
`UnsupportedOperation` instead of silently falling back.

---

## Adding a New Capability: Rerank (Design Sketch)

Rerank is intentionally not treated as an “OpenAI standard”; different
providers model it differently. The pattern remains the same:

1. **Core layer (`siumai-core`)**
   - `execution::rerank` defines:
     - `RerankInput` / `RerankOutput` / `RerankItem`.
     - `RerankRequestTransformer` / `RerankResponseTransformer`
       traits.

2. **Aggregator layer (`siumai` crate)**
   - `types::RerankRequest` / `RerankResponse` are the user-facing
     API.
   - Providers expose capability via `RerankCapability` on the
     client.
   - For OpenAI-compatible providers, the current implementation:
     - Maps `RerankRequest` into a minimal JSON body in
       `transform_rerank` (model / query / documents / top_n / ...).
     - Delegates provider-specific tweaks to
       `ProviderAdapter::transform_request_params` with
       `RequestType::Rerank`.

3. **Provider standard (optional per provider)**
   - For providers that have a rich rerank API (e.g. SiliconFlow),
     we can introduce a provider-specific “standard”:
     - Either in `siumai-std-*` (if reusable), or
     - Directly in `siumai-provider-*` as
       `SiliconflowRerankStandard`.
   - That standard would:
     - Map `RerankInput` → provider JSON.
     - Parse provider JSON → `RerankOutput`.
   - Aggregator would:
     - Detect `provider_id == "siliconflow"` (or use a dedicated
       `ProviderSpec`).
     - Delegate rerank operations to that standard rather than
       hand-building JSON.

This keeps the rerank integration aligned with the same patterns as
chat/embedding/image, while acknowledging that rerank is
provider-specific.

---

## Checklist for New Providers

When adding a new provider or capability, follow this checklist:

1. **Decide on the standard**
   - Reuse an existing standard (`std-openai`, `std-anthropic`,
     `std-gemini`) if the protocol matches.
   - If the provider has unique behaviour, consider a new standard
     crate or a provider-local standard module.

2. **Implement `CoreProviderSpec` (optional provider crate)**
   - Headers and base URLs.
   - Route selection (chat / embedding / image).
   - Choose standard transformers.

3. **Implement aggregator `ProviderSpec`**
   - Use `ProviderContext::to_core_context()` when delegating to
     external provider crates.
   - Use the appropriate `*_chat_request_to_core_input` helper.
   - Bridge streaming with `bridge_core_chat_transformers` +
     `map_core_stream_event_with_provider`.

4. **Wire the client**
   - Use `ChatExecutorBuilder`, `HttpChatExecutor`,
     `HttpEmbeddingExecutor`, `HttpImageExecutor` as appropriate.
   - Pass in interceptors, retry options, and model-level middlewares.

5. **Add tests and docs**
   - Include at least a small end-to-end test using mock HTTP.
   - Update `docs/architecture` / `docs/refactor` with any new
     patterns.

Following this pattern ensures new providers remain consistent with
the rest of Siumai and with the Vercel AI SDK–style architecture.
