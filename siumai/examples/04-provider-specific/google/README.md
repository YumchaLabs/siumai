# Google Provider Examples

This directory contains Google-specific examples across two closely related surfaces:

- Gemini (Google AI)
- Google Vertex

## Package tier

- `google/` in examples is a provider-owned package story
- Gemini examples are provider-owned and config-first by default
- Vertex examples live here only when they share the broader Google provider narrative; dedicated Vertex package guidance still lives under `provider_ext::google_vertex`

## Official Docs

- Gemini API overview: <https://ai.google.dev/gemini-api>
- Function calling: <https://ai.google.dev/gemini-api/docs/function-calling>
- Grounding / Google Search: <https://ai.google.dev/gemini-api/docs/grounding>
- Gemini REST reference: <https://ai.google.dev/api/rest/v1beta>
- Vertex AI Gemini reference: <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini>

## Auth and Base URL

### Google AI Gemini API

- Default base URL: `https://generativelanguage.googleapis.com/v1beta`
- Auth: `x-goog-api-key`
- Preferred construction: config-first `GeminiConfig::new(api_key)`
- Common env vars: `GEMINI_API_KEY`, sometimes `GOOGLE_API_KEY`

### Vertex AI

- Enterprise base URL:
  `https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google`
- Helper: `siumai::experimental::auth::vertex::google_vertex_base_url(project, location)`
- Express mode base URL: `https://aiplatform.googleapis.com/v1/publishers/google`
- Express mode may append `?key=...` when no Bearer auth is present
- Enterprise auth usually uses `Authorization: Bearer <token>`
- Token providers include:
  - `GeminiConfig::with_token_provider(Arc<dyn TokenProvider>)`
  - `AdcTokenProvider::default_client()` with `gcp`
- Current ADC-related env vars used by the implementation include:
  - `GOOGLE_OAUTH_ACCESS_TOKEN`
  - `GOOGLE_APPLICATION_CREDENTIALS`
  - `ADC_METADATA_URL` (test override)

## Capability Notes

These examples follow Siumai's current Gemini / Vertex conversion rules.
Unsupported tool combinations may be omitted intentionally to keep the Stable family surface predictable.

| Tool id | Helper | Notes |
|---|---|---|
| `google.google_search` | `siumai::hosted_tools::google::google_search()` | Gemini 2.x / 3.x uses `googleSearch`; Gemini 1.5 falls back to `googleSearchRetrieval`. |
| `google.file_search` | `siumai::hosted_tools::google::file_search()` | Requires Gemini 2.5-level support. |
| `google.code_execution` | `siumai::hosted_tools::google::code_execution()` | Availability follows Google model support. |
| `google.url_context` | `siumai::hosted_tools::google::url_context()` | Gemini 2.x / 3.x only. |
| `google.enterprise_web_search` | `siumai::hosted_tools::google::enterprise_web_search()` | Gemini 2.x / 3.x only. |
| `google.google_maps` | `siumai::hosted_tools::google::google_maps()` | Gemini 2.x / 3.x only. |
| `google.vertex_rag_store` | `siumai::hosted_tools::google::vertex_rag_store(..)` | Vertex-only; Gemini 2.x / 3.x only. |

## Examples

- `grounding.rs` - Google Search grounding
- `url_context.rs` - URL context tool
- `enterprise_web_search.rs` - enterprise web search
- `file_search.rs` - query-time file search
- `file_search-ext.rs` - file-search store management via provider extensions
- `logprobs.rs` - Google logprobs-oriented request shaping
- `vertex_chat.rs` - minimal Vertex chat via ADC (`--features "google gcp"`)
- `vertex_imagen_edit.rs` - Vertex Imagen edit / inpaint (`--features "google-vertex gcp"`)

## Vertex Imagen Notes

- Vertex Imagen uses the Vertex `:predict` endpoint
- Bearer auth is typically required in enterprise mode
- For editing/inpainting, set `ImageEditRequest.model` to an Imagen edit model such as `imagen-3.0-edit-001`
- Reference images and negative prompts can be passed via `providerOptions["vertex"]` when needed

## Recommended Reading Order

1. `grounding.rs`
2. `url_context.rs`
3. `file_search.rs`
4. `file_search-ext.rs`
5. `vertex_chat.rs`
6. `vertex_imagen_edit.rs`

## Notes

Google examples are provider-owned examples, not compat-preset examples. Prefer config-first here unless a specific file is explicitly demonstrating a Stable registry-first path.
