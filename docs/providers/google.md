# Google Gemini Provider Support

- Provider identity: `google`
- Technical platform: `gemini-api`
- Current portable family: Image
- Protocol / API mode: `gemini-interactions` / `interactions`
- API version: stable `v1`
- Evidence verified: 2026-08-08
- Public stability: stable
- Facade feature: `google`
- Provider crate: `siumai-provider-gemini`
- Protocol crate: `siumai-protocol-gemini`

`GeminiProvider` is the long-lived product owner. Public types use Gemini product terminology;
`google` remains the provider identity, facade feature, and typed provider-options namespace.
The configured provider is model-independent, synchronously constructed, and network-free.

```rust,no_run
use siumai::providers::google::models::GEMINI_3_1_FLASH_IMAGE;
use siumai::providers::google::{GeminiCredential, GeminiProvider};

let provider = GeminiProvider::builder(GeminiCredential::api_key(
    std::env::var("GEMINI_API_KEY")?,
))
.build()?;
let model = provider.image(GEMINI_3_1_FLASH_IMAGE)?;
# let _ = model;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Known model constants are dated hints, not an allowlist. Unknown future model IDs remain
constructible with protocol-baseline behavior; Siumai does not infer model-specific controls from
their names.

## Stable Interactions image contract

The current portable surface implements one `ImageModel` output per call through
`POST /v1/interactions`. `siumai-protocol-gemini` owns the wire schema, request encoding, terminal
status classification, image decoding, usage mapping, and response bounds. The provider crate owns
credentials, endpoint provenance, transport, model advisories, typed options, and support evidence.

Requests encode the current polymorphic `response_format` object. They never send deprecated
`outputs` or `response_mime_type` fields, and they do not substitute GenerateContent's
`response_modalities` field into Interactions.

Provider-owned controls remain typed:

- `GeminiImageAspectRatio` selects documented aspect ratios;
- `GeminiImageSize` selects `512`, `1K`, `2K`, or `4K` where the selected model supports it;
- `GeminiImageOptions` carries those values through the `google` namespace and `interactions` API
  mode.

Stable-v1 Interactions currently exposes JPEG as the explicit image MIME selection. An omitted
portable format leaves MIME selection to the service. Explicit PNG and other unsupported formats
fail before transport. Responses retain any bounded `image/*` MIME type returned by the service and
support both inline base64 and URI delivery.

## Endpoint ownership and replay

Only the provider-owned default endpoint receives verified Google Gemini evidence and the official
replay audience. Any endpoint supplied through `with_endpoint` or `with_base_url` is
caller-controlled even if its transport policy is labeled official. It must use an explicit custom
`ReplayDomain` and exposes only generic compatibility evidence.

This boundary does not model regions, commercial availability, routing, pricing, quota, or account
selection. Those remain host-application concerns.

## Current exclusions

The currently published portable slice is image generation. Language, embedding, buffered speech,
Files, Veo jobs, stored/background Interactions, and Live sessions are not claimed by this document
until their provider-owned implementations and deterministic fixtures land.

The former Imagen `models/*:predict` compatibility implementation remains deleted. Siumai does not
retain aliases for retired product paths.

## Evidence

| Scope | Official source |
|---|---|
| Stable Interactions request, response, status, step, usage, and response-format schema | [Interactions v1 API reference](https://ai.google.dev/api/interactions-api) |
| Current response-format migration and deprecated fields | [Interactions migration guide](https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026) |
| Image models, formats, aspect ratios, and resolution tiers | [Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation) |

Offline tests cover stable-v1 encoding, typed options, model-specific validation, direct and
Registry-erased execution, inline and URI response decoding, usage preservation, terminal status,
resource bounds, cancellation, endpoint provenance, replay-audience isolation, and sanitized
diagnostics. They do not perform live, credentialed, or billable calls.
