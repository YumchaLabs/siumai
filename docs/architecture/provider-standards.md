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

## Unified Reasoning Capabilities (Overview)

Several providers expose "reasoning" or "thinking" modes (OpenAI o1/o3,
Anthropic thinking, Gemini thinkingConfig, DeepSeek-style reasoning,
etc.). Siumai follows Vercel AI SDK's approach:

- Use provider-specific typed options (`providerOptions.*`) wherever
  possible.
- Offer a *unified* reasoning interface only for OpenAI-compatible
  providers, and keep its behaviour close to each provider's official
  docs.
- Avoid inventing undocumented HTTP fields; if a provider does not
  document a numeric `reasoning_budget`, Siumai will not send one.

At a glance:

| Provider family                       | User-facing API in Siumai                                                                                   | What you configure                                            | HTTP behaviour (simplified)                                                                                      | Notes                                                                                       |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| OpenAI (Chat / Responses, o1/o3)     | `OpenAiClient` + `OpenAiOptions` (`providerOptions.openai`)                                                | `reasoning_effort` (`low`/`medium`/`high`)                   | `ChatInput::extra["openai_reasoning_effort"]` → `reasoning_effort` in final JSON                                | Mirrors Vercel `providerOptions.openai.reasoningEffort`; no builder-level `reasoning()` yet |
| Anthropic (Claude Messages)          | `AnthropicClient` + `AnthropicOptions` (`providerOptions.anthropic`)                                       | `thinking_mode` (enabled + optional budget)                  | `ChatInput::extra["anthropic_thinking"]` → `thinking` object in final JSON                                      | Follows official `thinking` + prompt caching + JSON schema docs                             |
| Gemini (Gemini 1.5 / 2.x Chat)       | `GeminiClient` + generation helpers (`with_thinking_budget`, `with_dynamic_thinking`, `with_thinking_disabled`) | `thinking_config` (`thinkingBudget`, `includeThoughts`)      | `GenerationConfig.thinkingConfig` → `thinkingConfig` in `generationConfig` JSON                                  | No extra undocumented reasoning fields are added                                            |
| OpenAI-compatible providers (DeepSeek / SiliconFlow / Doubao / Qwen / OpenRouter, etc.) | `OpenAiCompatibleBuilder` (`LlmBuilder::new().deepseek()`, `.siliconflow()`, `.doubao()`, etc.)           | `.with_thinking(..)` / `.with_thinking_budget(..)` / `.reasoning(..)` / `.reasoning_budget(..)` | Stored in `OpenAiCompatibleConfig.provider_params` and mapped by each adapter to the provider's documented keys | Unified surface, provider-specific mapping; behaviour is kept conservative                  |

### Unified OpenAI-compatible reasoning API

For OpenAI-compatible providers, Siumai exposes a unified interface on
`OpenAiCompatibleBuilder`:

- `.with_thinking(bool)` / `.with_thinking_budget(u32)`
- `.reasoning(bool)` / `.reasoning_budget(i32)`

Internally, these map into `provider_params` which are then interpreted
by each provider adapter:

- **SiliconFlow**  
  - `with_thinking(true)` → `enable_thinking: true`  
  - `with_thinking_budget(8192)` → `thinking_budget: 8192` + `enable_thinking: true`
- **DeepSeek / OpenRouter**  
  - `reasoning(true)` → `enable_reasoning: true`  
  - `reasoning_budget(..)` stores a `reasoning_budget` value but does
    not send undocumented DeepSeek-specific numeric fields; enabling is
    controlled via `enable_reasoning`.
- **Doubao**  
  - `reasoning(true)` → internal `enable_thinking: true`, which the
    adapter converts into Doubao's documented `thinking` structure
    (e.g. `{ "type": "enabled" | "disabled" }`).
- **Other OpenAI-compatible providers**  
  - Default to `enable_reasoning` + optional `reasoning_budget` in
    `provider_params`, only if such fields are documented for that
    provider.

This mirrors Vercel AI SDK's unified `reasoning` surface for
OpenAI-compatible providers, while respecting each provider's official
API. The `provider-options-standard` design doc
(`docs/refactor/10-provider-options-standard.md`) describes how these
`provider_params` flow through to `ChatInput::extra` and, finally, to
provider JSON.

---

## Concrete Examples

### OpenAI-Specific Options (Reasoning / Web Search)

Siumai follows Vercel AI SDK’s pattern of “generic call options +
typed providerOptions” for OpenAI:

- Generic layer: `ChatRequest` / `ChatRequestBuilder`
- OpenAI layer: `OpenAiOptions` + `ProviderOptions::OpenAi`
- Core mapping: `openai_chat_request_to_core_input` /
  `build_responses_input`

Key pieces:

- `siumai::types::OpenAiOptions` lives in
  `siumai/src/types/provider_options/openai/mod.rs`.
- It mirrors Vercel’s `providerOptions.openai`:
  - `reasoning_effort: Option<ReasoningEffort>`
  - `service_tier: Option<ServiceTier>`
  - `web_search_options: Option<OpenAiWebSearchOptions>`
  - `responses_api: Option<ResponsesApiConfig>`
  - `modalities`, `audio`, `prediction`, etc.
- `ChatRequestBuilder::with_openai_options` wraps it into
  `ProviderOptions::OpenAi`.

The mapping rules:

- **Chat Completions** (`/chat/completions`):
  - `openai_chat_request_to_core_input` injects typed options into
    `ChatInput::extra["openai_*"]`, e.g.
    - `reasoning_effort` → `"openai_reasoning_effort"`
    - `service_tier` → `"openai_service_tier"`
    - `modalities` → `"openai_modalities"`
    - `audio` → `"openai_audio"`
    - `prediction` → `"openai_prediction"`
    - `web_search_options` → `"openai_web_search_options"`

- **Responses API** (`/responses`):
  - `build_responses_input` constructs `ResponsesInput` from
    `ChatRequest`:
    - Maps messages via `OpenAiResponsesRequestTransformer`.
    - Fills `ResponsesInput.extra` with:
      - `stream`, `seed`, `max_output_tokens`
      - `tools` / `tool_choice`
      - `responses_api` fields:
        `instructions`, `response_format`, `previous_response_id`,
        `truncation`, `include`, `store`, `max_tool_calls`, etc.
      - `reasoning_effort` / `service_tier` (typed enums).
      - `modalities`, `audio`, `prediction`.
      - `web_search_options` (typed `OpenAiWebSearchOptions`).

From a user’s perspective this matches Vercel’s design:

```rust,no_run
use siumai::prelude::*;
use siumai::types::{
    ChatRequest, OpenAiOptions,
    provider_options::openai::{
        ReasoningEffort, OpenAiWebSearchOptions, WebSearchLocation,
        ResponsesApiConfig,
    },
};

async fn openai_responses_with_reasoning_and_websearch(
    client: siumai::providers::openai::OpenAiClient,
) -> Result<(), Box<dyn std::error::Error>> {
    // Configure web search options (similar to Vercel providerOptions.openai.webSearchOptions)
    let web_search = OpenAiWebSearchOptions::new()
        .with_context_size("high")
        .with_user_location(
            WebSearchLocation::approximate()
                .with_country("US")
                .with_city("San Francisco"),
        );

    // Configure Responses API + reasoningEffort, just like Vercel
    let opts = OpenAiOptions::new()
        .with_reasoning_effort(ReasoningEffort::Medium)
        .with_web_search_options(web_search)
        .with_responses_api(
            ResponsesApiConfig::new()
                .with_background(false),
        );

    let req = ChatRequest::new(vec![user!("What’s new in Rust 1.80?")])
        .with_openai_options(opts);

    let resp = client.chat_request(req).await?;
    println!("answer = {}", resp.text());

    if let Some(meta) = resp.openai_metadata() {
        if let Some(reasoning_tokens) = meta.reasoning_tokens {
            println!("reasoning tokens used = {}", reasoning_tokens);
        }
    }
    Ok(())
}
```

This keeps all OpenAI-specific JSON/fields at the OpenAI layer,
while the aggregator only sees `ChatRequest` + `ProviderOptions::OpenAi`
and core standards (`ResponsesInput` / `ChatInput`).

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

- **Anthropic-specific options**:
  - `AnthropicOptions` in
    `siumai/src/types/provider_options/anthropic.rs` mirrors Vercel’s
    `providerOptions.anthropic`:
    - `prompt_caching: Option<PromptCachingConfig>`
    - `thinking_mode: Option<ThinkingModeConfig>`
    - `response_format: Option<AnthropicResponseFormat>`
  - `ChatRequestBuilder::with_anthropic_options` wraps it into
    `ProviderOptions::Anthropic`.
  - `anthropic_like_chat_request_to_core_input` maps typed options into
    `ChatInput::extra` with Anthropic-style keys:
    - `thinking_mode` → `"anthropic_thinking"` (later renamed to
      `thinking` in the final JSON by `siumai-std-anthropic`).
    - `response_format` → `"anthropic_response_format"`.
    - `prompt_caching` → `"anthropic_prompt_caching"`.

  From a user’s perspective this matches Vercel’s design of
  `providerOptions.anthropic.thinking` / `promptCaching` /
  `responseFormat`:

  ```rust,no_run
  use siumai::prelude::*;
  use siumai::types::{
      ChatRequest, AnthropicOptions,
      provider_options::anthropic::{
          PromptCachingConfig, AnthropicCacheControl, AnthropicCacheType,
          ThinkingModeConfig, AnthropicResponseFormat,
      },
  };

  async fn anthropic_with_thinking_and_prompt_caching(
      client: siumai::providers::anthropic::AnthropicClient,
  ) -> Result<(), Box<dyn std::error::Error>> {
      // Enable extended thinking with an explicit budget
      let thinking = ThinkingModeConfig {
          enabled: true,
          thinking_budget: Some(10_000),
      };

      // Enable prompt caching on the first user message
      let caching = PromptCachingConfig {
          enabled: true,
          cache_control: vec![AnthropicCacheControl {
              cache_type: AnthropicCacheType::Ephemeral,
              message_index: 0,
          }],
      };

      // Configure structured output as JSON schema
      let response_format = AnthropicResponseFormat::JsonSchema {
          name: "Answer".to_string(),
          schema: serde_json::json!({
              "type": "object",
              "properties": {
                  "summary": { "type": "string" }
              },
              "required": ["summary"]
          }),
          strict: true,
      };

      let opts = AnthropicOptions::new()
          .with_thinking_mode(thinking)
          .with_prompt_caching(caching)
          .with_json_schema("Answer", serde_json::json!({}), true);

      let req = ChatRequest::new(vec![user!("Explain Rust ownership.")])
          .with_anthropic_options(opts);

      let resp = client.chat_request(req).await?;
      println!("answer = {}", resp.text());
      Ok(())
  }
  ```

  All Anthropic-specific JSON (thinking / response_format / prompt
  caching) is constructed in `anthropic_like_chat_request_to_core_input`
  + `siumai-std-anthropic`; the aggregator only coordinates
  `ChatRequest` + `ProviderOptions::Anthropic` and the standard
  transformers.


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

- **Gemini-specific options**:
  - `GeminiOptions` in
    `siumai/src/types/provider_options/gemini.rs` mirrors Vercel’s
    `providerOptions.google` for Gemini models:
    - `code_execution: Option<CodeExecutionConfig>` → enables the
      `codeExecution` tool.
    - `search_grounding: Option<SearchGroundingConfig>` → enables
      `googleSearchRetrieval.dynamicRetrievalConfig`.
    - `file_search: Option<FileSearchConfig>` → best-effort File Search
      tool wiring (non-standard schema).
    - `response_mime_type: Option<String>` → sets
      `generationConfig.responseMimeType`.
  - `ChatRequestBuilder::with_gemini_options` wraps it into
    `ProviderOptions::Gemini`.
  - `gemini_like_chat_request_to_core_input` maps typed options into
    `ChatInput::extra`:
    - `code_execution` → `"gemini_code_execution"`.
    - `search_grounding` → `"gemini_search_grounding"`.
    - `file_search` → `"gemini_file_search"`.
    - `response_mime_type` → `"gemini_response_mime_type"`.
  - `GeminiDefaultChatAdapter` in
    `siumai-std-gemini::gemini::chat` consumes these keys and injects
    official Gemini JSON fields:
    - `gemini_code_execution.enabled == true` →
      `tools += { "codeExecution": {} }`.
    - `gemini_search_grounding.enabled == true` →
      `tools += { "googleSearchRetrieval": { "dynamicRetrievalConfig": {...} } }`.
    - `gemini_file_search.file_search_store_names != []` →
      `tools += { "file_search": { "file_search_store_names": [...] } }`
      (experimental, not an official schema).
    - `gemini_response_mime_type` → `generationConfig.responseMimeType`.

  This matches the way Vercel models Gemini-specific behaviour via
  `providerOptions.google.*` while keeping the aggregator free of JSON
  details. A typical usage mirrors Vercel’s pattern:

  ```rust,no_run
  use siumai::prelude::*;
  use siumai::types::{
      ChatRequest, GeminiOptions,
      provider_options::gemini::{
          CodeExecutionConfig, SearchGroundingConfig, DynamicRetrievalConfig,
      },
  };

  async fn gemini_with_code_execution_and_search(
      client: siumai::providers::gemini::GeminiClient,
  ) -> Result<(), Box<dyn std::error::Error>> {
      // Enable code execution tool
      let code = CodeExecutionConfig { enabled: true };

      // Enable Google search grounding with dynamic retrieval
      let search = SearchGroundingConfig {
          enabled: true,
          dynamic_retrieval_config: Some(DynamicRetrievalConfig {
              mode: crate::types::provider_options::gemini::DynamicRetrievalMode::ModeDynamic,
              dynamic_threshold: Some(0.5),
          }),
      };

      let opts = GeminiOptions::new()
          .with_code_execution(code)
          .with_search_grounding(search)
          .with_response_mime_type("application/json");

      let req = ChatRequest::new(vec![user!("Explain Rust lifetimes.")])
          .with_gemini_options(opts);

      let resp = client.chat_request(req).await?;
      println!("answer = {}", resp.text());
      Ok(())
  }
  ```

  For structured outputs and thinking-specific configuration, the
  Gemini client also exposes `GenerationConfig` helpers:

  - `GeminiClient::with_json_schema(...)` /
    `with_enum_schema(...)` → `generationConfig.responseMimeType` +
    `responseSchema`.
  - `GeminiClient::with_thinking_budget(...)` /
    `with_dynamic_thinking()` /
    `with_thinking_disabled()` → `generationConfig.thinkingConfig`
    with `thinkingBudget` / `includeThoughts`.

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
