# Google (Gemini) provider-specific examples

This folder contains **Gemini (Google AI / Vertex AI)** examples (and a separate **Google Vertex** Imagen example)
that align with the Vercel AI SDK model and tool boundaries.

## Official docs

- Gemini API overview: https://ai.google.dev/gemini-api
- Function calling: https://ai.google.dev/gemini-api/docs/function-calling
- Grounding (Google Search): https://ai.google.dev/gemini-api/docs/grounding
- REST (v1beta): https://ai.google.dev/api/rest/v1beta
- Vertex AI Gemini: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini

## Auth & base_url (Google AI vs Vertex)

- **Google AI Gemini API** (default base URL: `https://generativelanguage.googleapis.com/v1beta`)
  - Auth: `x-goog-api-key` (set via `.api_key(...)` or `SIUMAI_API_KEY`)
- **Vertex AI** (recommended base URL: `https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google`)
  - Set base URL via `SiumaiBuilder::base_url_for_vertex(project, location, "google")`
  - Auth: `Authorization: Bearer <token>` via:
    - `.with_gemini_token_provider(Arc<dyn TokenProvider>)` (`feature = "google"` or `feature = "google-vertex"`)
    - `.with_gemini_adc()` (`feature = "gcp"` + (`google` or `google-vertex`))
  - ADC env vars (current implementation): `GOOGLE_OAUTH_ACCESS_TOKEN`, `GOOGLE_APPLICATION_CREDENTIALS`, `ADC_METADATA_URL` (test override)
  - Model id: prefer bare ids (e.g. `gemini-2.0-flash`), but resource-style ids are accepted (e.g. `models/gemini-2.0-flash`, `publishers/google/models/gemini-2.0-flash`)

## Capabilities & model constraints (best-effort)

These constraints are enforced by Siumai’s Gemini protocol conversion logic. Unsupported tools are
silently omitted to keep the unified surface stable.

| Tool id | Helper | Notes / constraints |
|---|---|---|
| `google.google_search` | `siumai::hosted_tools::google::google_search()` | On Gemini **2.x/3.x** maps to `googleSearch`. On Gemini **1.5** maps to legacy `googleSearchRetrieval`. |
| `google.file_search` | `siumai::hosted_tools::google::file_search()` | Requires Gemini **2.5** (Gemini 2.0 models will ignore it). |
| `google.code_execution` | `siumai::hosted_tools::google::code_execution()` | Generally supported where Google exposes the tool; follow model-specific docs. |
| `google.url_context` | `siumai::hosted_tools::google::url_context()` | Gemini **2.x/3.x** only. |
| `google.enterprise_web_search` | `siumai::hosted_tools::google::enterprise_web_search()` | Gemini **2.x/3.x** only. |
| `google.google_maps` | `siumai::hosted_tools::google::google_maps()` | Gemini **2.x/3.x** only. |
| `google.vertex_rag_store` | `siumai::hosted_tools::google::vertex_rag_store(..)` | Vertex-only; Gemini **2.x/3.x** only. |

## Examples

- `grounding.rs`: Google Search grounding (provider-hosted tool).
- `url_context.rs`: URL context tool.
- `enterprise_web_search.rs`: Enterprise web search tool.
- `file_search.rs`: File search tool usage (query-time retrieval).
- `file_search-ext.rs`: File Search Stores (resource management; provider extension API).
- `vertex_chat.rs`: Minimal Vertex AI chat via ADC (`--features "google gcp"`).
- `vertex_imagen_edit.rs`: Vertex AI Imagen edit/inpaint with mask + reference images (`--features "google-vertex gcp"`).

## Vertex Imagen notes

- Vertex Imagen uses the Vertex `:predict` endpoint and requires Bearer auth (ADC/service account).
- For editing/inpainting, set `ImageEditRequest.model` to an Imagen edit model (e.g. `imagen-3.0-edit-001`).
- Reference images and negative prompts can be passed via `providerOptions["vertex"]` (preferred) or `extra_params`.
