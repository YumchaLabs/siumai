# Google Image Provider Support

- Provider identity: `google`
- Technical platform: `gemini-api`
- API mode: `interactions-image`
- Evidence verified: 2026-08-06
- Public stability: experimental
- Facade feature: `google`
- Owning crate: `siumai-provider-gemini`

This provider is intentionally narrower than the Gemini product. It implements image generation
through the beta Interactions API and does not claim Gemini language, Live, files, or Vertex media
support.

## Construction and model access

`GoogleImageProvider` is long-lived, model-independent, synchronously configured, and network-free:

```rust,no_run
use siumai::providers::google::{GoogleCredential, GoogleImageProvider};
use siumai::providers::google::models::GEMINI_3_1_FLASH_IMAGE;

let provider = GoogleImageProvider::builder(GoogleCredential::api_key(
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

## Image contract

The provider implements the shared `ImageModel` family with one requested output per call. Portable
PNG and JPEG format requests map to the Interactions image response format. Arbitrary pixel
dimensions are rejected because the remote API uses named resolution tiers instead.

Google-specific controls remain typed provider options:

- `GoogleImageAspectRatio` selects the documented aspect-ratio values;
- `GoogleImageSize` selects `512`, `1K`, `2K`, or `4K` where the chosen model supports it;
- `GoogleImageOptions` carries those values through the `google` namespace and
  `interactions-image` API mode.

The provider validates documented differences for the current Flash, Flash Lite, and Pro image
models before dispatch. Future IDs accept only protocol-safe baseline behavior without a named
capability claim.

## Deliberate exclusions

The former Imagen `models/*:predict` implementation is removed. Google marks the Imagen 4 models
on that API as deprecated and scheduled to stop functioning on 2026-08-17. Siumai does not keep a
compatibility alias for an endpoint at end of life.

This crate also does not expose Gemini language generation, multimodal conversation, Live,
long-running media jobs, files, or a general Interactions client. Those capabilities may return only
as independently designed provider-owned APIs with focused evidence and lifecycle contracts.

## Evidence

All named support was checked against official Google documentation on 2026-08-06. The local Vercel
AI SDK checkout is secondary design and fixture evidence only.

| Scope | Official source |
|---|---|
| Image generation, current models, formats, aspect ratios, and resolution tiers | [Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation) |
| Interactions request, response, status, step, image block, and usage schema | [Interactions API](https://ai.google.dev/api/interactions-api) |
| Imagen 4 deprecation and migration guidance | [Migrate from Imagen to Gemini native image generation](https://ai.google.dev/gemini-api/docs/imagen-to-gemini) |

Offline tests cover typed options, model-specific validation, direct and Registry-erased execution,
request encoding, base64 and URI response decoding, usage preservation, cancellation, replay
safety, and sanitized diagnostics. They do not perform live, credentialed, or billable calls.
