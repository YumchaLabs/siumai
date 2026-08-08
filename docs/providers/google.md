# Google Gemini Provider Support

- Provider identity: `google`
- Technical platform: `gemini-api`
- Current portable families: Language and Image
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
use siumai::providers::google::models::{GEMINI_3_1_FLASH_IMAGE, GEMINI_3_6_FLASH};
use siumai::providers::google::{GeminiCredential, GeminiProvider};

let provider = GeminiProvider::builder(GeminiCredential::api_key(
    std::env::var("GEMINI_API_KEY")?,
))
.build()?;
let language = provider.language(GEMINI_3_6_FLASH)?;
let image = provider.image(GEMINI_3_1_FLASH_IMAGE)?;
# let _ = (language, image);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Known model constants are dated hints, not an allowlist. Unknown future model IDs remain
constructible with protocol-baseline behavior; Siumai does not infer model-specific controls from
their names.

## Stable Interactions language contract

`provider.language(model)` and the explicit `provider.interactions(model)` entry point both create
the stable-v1 Interactions `LanguageModel`. Direct and streaming calls use `POST /v1/interactions`,
normalize caller-executed function arguments to one checked JSON object, preserve provider-native
thought and function-call replay as bounded opaque items, and require every established stream to
settle exactly once.

`GeminiLanguageModel::generate_native` retains the complete direct Interactions resource alongside
its canonical projection. The portable `LanguageModel::generate` path returns that same canonical
projection, while unknown output content needed for replay remains available as bounded
provider-native content.

The portable language projection supports role-safe text and media input, assistant replay, local
function tools, structured response formats, stop sequences, output-token limits, seeds, usage,
reasoning summaries, and provider metadata. Provider-owned controls remain typed through
`GeminiInteractionsOptions`:

- `GeminiInteractionStorage` makes storage opt-in; `store: false` is the default;
- `GeminiThinkingLevel` selects the stable Interactions thinking levels;
- `GeminiThinkingSummaries` controls provider thought summaries.

Stable Interactions does not currently expose portable temperature or top-p fields in its v1
schema, so explicit portable values fail before transport instead of being silently discarded.
Unknown or provider-only output steps remain available through bounded provider-native replay data.

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

The currently published portable slices are Interactions language and image generation.
GenerateContent mode, embedding, buffered speech, Files, Veo jobs, stored/background Interactions,
and Live sessions are not claimed by this document until their provider-owned implementations and
deterministic fixtures land.

The former Imagen `models/*:predict` compatibility implementation remains deleted. Siumai does not
retain aliases for retired product paths.

## Evidence

| Scope | Official source |
|---|---|
| Stable Interactions request, response, status, step, usage, and response-format schema | [Interactions v1 API reference](https://ai.google.dev/api/interactions-api) |
| Current response-format migration and deprecated fields | [Interactions migration guide](https://ai.google.dev/gemini-api/docs/interactions-breaking-changes-may-2026) |
| Image models, formats, aspect ratios, and resolution tiers | [Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation) |

Offline tests cover stable-v1 language and image encoding, typed options, model-specific
validation, direct and streaming language settlement, Registry-erased execution, canonical tool
arguments, native replay parity, inline and URI media decoding, usage preservation, terminal
status, resource bounds, cancellation, endpoint provenance, replay-audience isolation, and
sanitized diagnostics. They do not perform live, credentialed, or billable calls.
