# Google Gemini Provider Support

- Provider identity: `google`
- Technical platform: `gemini-api`
- Primary language mode: stable-v1 Interactions
- Secondary language mode: stable-v1 Generate Content, upstream `Legacy`
- Portable families: Language, Embedding, Image, and buffered Speech
- Provider-native surfaces: Files metadata lifecycle and Veo long-running jobs
- Evidence verified: 2026-08-08
- Facade feature: `google`
- Provider crate: `siumai-provider-gemini`
- Protocol crate: `siumai-protocol-gemini`

`GeminiProvider` is the long-lived product owner. Public types use Gemini product terminology;
`google` remains the provider identity, facade feature, and typed provider-options namespace. The
configured provider is model-independent, synchronously constructed, network-free, cheap to clone,
and backed by one shared transport runtime.

```rust,no_run
use siumai::providers::google::models::{
    GEMINI_3_1_FLASH_IMAGE, GEMINI_3_6_FLASH, GEMINI_EMBEDDING_001,
};
use siumai::providers::google::{GeminiCredential, GeminiProvider};

let provider = GeminiProvider::builder(GeminiCredential::api_key(
    std::env::var("GEMINI_API_KEY")?,
))
.build()?;

let language = provider.language(GEMINI_3_6_FLASH)?;
let legacy_generate_content = provider.generate_content(GEMINI_3_6_FLASH)?;
let embedding = provider.embedding(GEMINI_EMBEDDING_001)?;
let image = provider.image(GEMINI_3_1_FLASH_IMAGE)?;
let files = provider.files();
let veo = provider.veo();
# let _ = (language, legacy_generate_content, embedding, image, files, veo);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Known model constants are dated hints, not an allowlist. Unknown future model IDs remain
constructible with protocol-baseline behavior; Siumai does not infer capabilities or defaults from
model-name patterns.

## Language API modes

`provider.language(model)` and `provider.interactions(model)` select stable-v1 Interactions. This is
the default `LanguageModelProvider` and Registry registration. Direct and streaming calls use
`POST /v1/interactions`, normalize caller-executed function arguments to one checked JSON value,
preserve provider-native replay as bounded opaque items, and require exactly one terminal outcome.

`provider.generate_content(model)` selects the distinct `gemini-generate-content` /
`generate-content` API mode. It uses the still-published stable-v1 `generateContent` and
`streamGenerateContent` operations, but Google currently labels the product path "Generate Content
API (Legacy)." Siumai records stable protocol maturity and Legacy upstream support independently.
It is never selected implicitly; callers may register it through
`provider.generate_content_registration()` when a route intentionally targets that mode.

Generate Content supports role-safe text and media input, assistant replay, local function tools,
structured output through `generationConfig.responseFormat.text`, sampling controls, thinking,
usage, and direct/stream parity. A finish reason remains provisional until trailing usage frames and
clean EOF have been processed. `[DONE]` is not treated as a valid Google settlement event.

Provider-owned controls remain typed:

- `GeminiInteractionsOptions` controls Interactions storage and thinking behavior;
- `GeminiGenerateContentOptions` controls service tier, storage, top-k, and thinking behavior;
- `GeminiLanguageModel::generate_native` and `GeminiGenerateContentModel::generate_native` retain
  the complete direct provider resource beside the canonical response.

## Portable embedding, image, and speech

Text embedding uses stable-v1 `embedContent` and `batchEmbedContents`. Requests use the nested
`embedContentConfig` shape, default `autoTruncate` to `false`, preserve input ordering, and reject
known unsupported combinations before transport. The current claim is text embedding only; it is
not a claim for Gemini's complete multimodal embedding product surface.

Image generation uses stable-v1 Interactions. Requests encode the current polymorphic
`response_format` object and never send deprecated `outputs` or `response_mime_type` fields.
`GeminiImageOptions` exposes documented aspect ratio and image-size controls while the portable
adapter retains bounded inline or URI image outputs.

Buffered speech uses the current v1beta Interactions audio response path. The portable
`SpeechModel` slice requires an explicit voice and currently accepts Gemini's default raw 24 kHz PCM
output. Numeric speed, explicit format, and language override fail before transport because the
implemented upstream path does not expose equivalent controls. Preview upstream maturity is
recorded independently from Siumai's experimental public stability.

## Provider-native Files and Veo

`provider.files()` implements stable-v1 File metadata `get`, `list`, and `delete`. Upload and GCS
registration remain deferred because they have distinct upload/OAuth lifecycle and credential
requirements. Download URIs and provider error details are redacted from default diagnostics.

`provider.veo()` implements the current provider-owned Veo 3.1 preview slice:

- text or inline-image submission through `predictLongRunning`;
- replay-bound typed operation references;
- explicit status retrieval;
- pending, succeeded, RAI-filtered, and failed states;
- no hidden polling, automatic download, or generic cross-provider video trait.

Submit requests are never automatically replayed because duplicate submission may create another
billable job. Status reads are semantically idempotent. Operation references cannot be reused across
different replay audiences.

## Endpoint ownership and replay

Only the provider-owned default endpoint receives verified Google Gemini evidence, dated model
advice, official native claims, and the official replay audience. Any endpoint supplied through
`with_endpoint` or `with_base_url` is caller-controlled even if its transport policy is labeled
official. It requires an explicit custom `ReplayDomain` and exposes only generic compatibility
claims.

This boundary does not model regions, projects, commercial availability, routing, pricing, quota,
or account entitlement. Those remain host-application concerns.

## Deliberately deferred

The current provider does not claim:

- stored/background Interactions lifecycle resources;
- Live sessions or ephemeral tokens;
- File upload or GCS registration;
- Veo automatic polling, download, extension, seed, or broad reference-image workflows;
- a provider-neutral video-job trait;
- broad Vertex Gemini/media coverage.

The former Imagen `models/*:predict` compatibility implementation remains deleted. Siumai does not
retain aliases for retired product paths.

## Evidence

| Exact slice | Official source |
|---|---|
| Stable Interactions request, response, status, step, usage, and response format | [Interactions v1 API reference](https://ai.google.dev/api/interactions-api) |
| Stable-v1 Generate Content operations and current Legacy product posture | [Generate Content API (Legacy)](https://ai.google.dev/gemini-api/docs/generate-content/text-generation) |
| Stable-v1 text embedding | [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings) |
| Stable Interactions image generation | [Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation) |
| Interactions speech generation | [Gemini speech generation](https://ai.google.dev/gemini-api/docs/speech-generation) |
| File metadata lifecycle | [Gemini Files](https://ai.google.dev/gemini-api/docs/files) |
| Veo submit and long-running operation lifecycle | [Veo video generation](https://ai.google.dev/gemini-api/docs/veo) |
| Stable-v1 operation discovery, including Generate Content and Files | [Gemini API v1 discovery](https://generativelanguage.googleapis.com/$discovery/rest?version=v1) |

Deterministic offline fixtures cover direct and streaming language settlement, trailing usage,
canonical tool arguments, replay parity, text embedding, image generation, buffered speech, Files
get/list/delete, Veo submit/status, endpoint provenance, replay isolation, resource bounds, and
sanitized diagnostics. They do not perform live, credentialed, or billable calls.
