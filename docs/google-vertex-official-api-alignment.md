# Google Vertex Official API Alignment (Gemini + Embeddings + Imagen)

This document records **official API correctness checks** for Google Vertex AI APIs that Siumai supports,
and how Siumai maps/validates the same behavior (during the Alpha.5 fearless refactor).

It complements:

- `docs/vercel-ai-fixtures-alignment.md` (fixture-driven parity vs Vercel AI SDK)
- `docs/provider-implementation-alignment.md` (global provider audit checklist)
- `docs/core-trio-module-alignment.md` (module mapping for the core trio)

## Sources (Google Cloud docs)

Vertex Gemini / Generative AI (REST references):

- Enterprise REST (publisher models):
  - `models.generateContent` (v1beta1): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.publishers.models/generateContent>
  - `models.streamGenerateContent` (v1beta1): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/projects.locations.publishers.models/streamGenerateContent>
- Express mode REST (publisher models):
  - `publishers.models.generateContent` (v1): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/express-mode/rest/v1/publishers.models/generateContent>
  - `publishers.models.streamGenerateContent` (v1): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/express-mode/rest/v1/publishers.models/streamGenerateContent>

Embeddings:

- Text embeddings API: <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings>

Imagen:

- Generate images: <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/generate-images>
- Edit images (insert objects): <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/image/edit-insert-objects>
- Imagen REST / parameters reference: <https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/imagen-api>

## Local access note (contributors)

In some regions, `docs.cloud.google.com` may be unstable.
This repo’s contributors commonly use a local HTTP proxy:

```bash
set HTTP_PROXY=http://127.0.0.1:10809
set HTTPS_PROXY=http://127.0.0.1:10809
```

## Siumai implementation (where to compare)

Vertex provider:

- Provider wiring:
  - `siumai-provider-google-vertex/src/providers/vertex/*`
- Base URL / mode selection:
  - `siumai-provider-google-vertex/src/providers/vertex/builder.rs`

Gemini via Vertex (GenerateContent):

- Chat standard (reuses Gemini protocol mapping + Vertex auth rules):
  - `siumai-provider-google-vertex/src/standards/vertex_generative_ai.rs`
- Shared Gemini mapping:
  - `siumai-protocol-gemini/src/standards/gemini/*`

Embeddings via Vertex (`:predict`):

- `siumai-provider-google-vertex/src/standards/vertex_embedding.rs`

Imagen via Vertex (`:predict`):

- Request/response mapping:
  - `siumai-provider-google-vertex/src/standards/vertex_imagen.rs`
- Provider options schema (Vercel-aligned `providerOptions.vertex`):
  - `siumai-provider-google-vertex/src/provider_options/vertex/imagen.rs`

## Base URL + endpoint patterns (official)

The official REST references model the “model name” as a fully-qualified resource, and then append
the RPC-style suffix (`:generateContent`, `:streamGenerateContent`, `:predict`) to it.

### Gemini (GenerateContent / StreamGenerateContent)

Official resource formats (docs):

- Publisher model format: `publishers/google/models/*`
- Enterprise publisher model format (common in practice): `projects/{project}/locations/{location}/publishers/google/models/{model}`

### Siumai mapping

Siumai uses a “base URL prefix + relative model id” composition, which produces an equivalent URL:

- Enterprise mode (project + location):
  - `https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/publishers/google`
  - Then Siumai appends:
    - Non-stream: `/models/{model}:generateContent`
    - Stream: `/models/{model}:streamGenerateContent` (Siumai uses SSE mode; see notes below)
- Express mode (API key):
  - `https://aiplatform.googleapis.com/v1/publishers/google`
  - Then Siumai appends the same suffixes under `/models/{model}:...`

This matches Vercel AI SDK’s base URL behavior and is locked down by fixtures.

## Authentication + required headers (official)

From official Vertex AI examples:

- `Authorization: Bearer $(gcloud auth print-access-token)` (OAuth2 / ADC)
- `Content-Type: application/json; charset=utf-8`

### Siumai mapping

- Enterprise mode:
  - `GoogleVertexConfig.token_provider` can inject `Authorization: Bearer ...` when missing.
  - Custom headers are merged and can override auth (advanced use-cases).
- Express mode:
  - Siumai supports a Vercel-aligned API-key flow by appending `?key=...` to the URL when no
    `Authorization` header is present.

## Streaming (Vertex Gemini)

Official REST references describe `streamGenerateContent` as a server-streaming endpoint.

### Siumai mapping

For gateway/proxy use-cases and Vercel parity, Siumai consumes and emits Vertex streaming
as SSE frames (Gemini-style `text/event-stream`).

- Implementation:
  - URL builder: `siumai-provider-google-vertex/src/standards/vertex_generative_ai.rs`
  - Streaming parser/serializer: `siumai-protocol-gemini/src/standards/gemini/streaming.rs`

**Note:** Siumai uses `?alt=sse` for Vertex Gemini streaming (Vercel-aligned). This query flag is
not described in the Vertex REST reference pages above, but is used by the wider Gemini REST docs
and is supported by the service in practice.

## Imagen (generation + edit/mask/referenceImages)

Official Imagen on Vertex uses the PredictionService `:predict` endpoint:

- `POST https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict`

The official image editing docs include:

- `instances[].referenceImages[]` for supplying source images, masks, and optional additional references
- Mask configuration under `maskImageConfig` (e.g. `maskMode`, `dilation`)
- Edit mode controls such as `editMode` and `editConfig.baseSteps`

### Siumai mapping

- URL format:
  - `siumai-provider-google-vertex/src/standards/vertex_imagen.rs` builds:
    - `{base_url}/models/{model}:predict`
  - Model id normalization accepts `models/...` and full resource-style ids (best-effort).
- Request body:
  - Generation uses `{ instances: [{ prompt }], parameters: { sampleCount, ... } }`
  - Editing uses `{ instances: [{ prompt, referenceImages }], parameters: { editMode, ... } }`
  - Mask is emitted as `REFERENCE_TYPE_MASK` with `maskImageConfig` and defaults to
    `MASK_MODE_USER_PROVIDED` (Vercel-aligned).
  - Additional `referenceImages` can be provided via `extra_params` or `providerOptions.vertex`.
- Tests/fixtures:
  - `siumai/tests/vertex_imagen*_fixtures_alignment_test.rs`

## Status

- Google Vertex (Gemini + Embeddings + Imagen) is treated as **Green** for parity and “practical official correctness”
  (auth, base URL composition, endpoint patterns, Imagen edit/mask/referenceImages mapping).

