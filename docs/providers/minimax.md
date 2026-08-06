# MiniMax Provider Support

- Provider identity: `minimax`
- Technical platform: `minimax-api`
- Evidence verified: 2026-08-06
- Facade feature: `minimax`
- Owning crate: `siumai-provider-minimax`

This document records the capabilities implemented by the current MiniMax provider. It describes
technical protocol and resource support, not account entitlement, pricing, quota, region-specific
availability, or a promise that every model is enabled for every caller.

## Construction and identity

`MinimaxProvider` is a long-lived, model-independent provider. Construction is synchronous and
network-free:

```rust,no_run
use siumai::providers::minimax::{MinimaxCredential, MinimaxProvider};

let provider = MinimaxProvider::builder(MinimaxCredential::api_key(
    std::env::var("MINIMAX_API_KEY")?,
))
.build()?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Official endpoints require a bearer credential. Separate Messages, OpenAI-compatible, and native
resource base URLs may be supplied for a gateway or test server, but they are technical addressing
inputs. Siumai does not define a MiniMax region enum or maintain a region-by-model availability
catalog.

Model identifiers are open. The constants under `siumai::providers::minimax::models` are dated
discovery hints and exact-match policy evidence, not an allowlist. Unknown future IDs remain
constructible when the selected protocol can encode the request safely; Siumai does not infer
model-specific controls for unknown IDs.

## Language API modes

| Siumai constructor | Remote API mode | Fidelity | Public stability | Current boundary |
|---|---|---|---|---|
| `language(model)` / `messages(model)` | Anthropic-compatible Messages | Verified-compatible | Stable | Recommended default; generation and streaming; typed thinking and service tier; fixed ephemeral prompt-cache annotations; MiniMax-M3 image/video input |
| `chat_completions(model)` | OpenAI-compatible Chat Completions | Verified-compatible | Experimental | Explicit opt-in; generation and streaming; typed thinking and service tier; `reasoning_split`; MiniMax-M3 image/video input |
| `responses(model)` | OpenAI-compatible Responses | Verified-compatible | Experimental | Explicit bounded subset; generation and streaming; text output; typed reasoning, service tier, prompt-cache key, and metadata; MiniMax-M3 image/video input |

The default Registry registration also selects Messages. Chat Completions and Responses have
separate provider-owned registrations so a host must opt into those modes deliberately.

The shared `LanguageRequest` contains portable semantics only. Mode-specific call behavior is
attached through `CallOptions` using one of:

- `MinimaxMessagesOptions`;
- `MinimaxChatCompletionsOptions`;
- `MinimaxResponsesOptions`.

Prompt-cache breakpoints for Messages use typed `MinimaxMessageCache`, `MinimaxContentCache`, or
`MinimaxToolCache` annotations beside the semantic node they modify. The codec emits MiniMax's
fixed ephemeral cache marker and does not invent a configurable TTL.

The provider rejects unsupported combinations before network submission. Examples include a
foreign or wrong-mode provider option, unsupported structured output, invalid tool-choice modes,
out-of-range known-model output limits, and media on a model without verified media support.

## Provider-native resources

The configured provider exposes native resources directly. These are typed provider-owned APIs,
not claims that the resource lifecycle is portable across vendors. They are exposed through the
documented provider-owned stable namespace; that stability does not promote them into a neutral
model-family contract.

| Entry point | Implemented operations | Verified boundary |
|---|---|---|
| `provider.files()` | Upload, list, retrieve metadata, download content, delete | Operation-specific purpose enums, positive signed-64-bit file IDs, bounded multipart input, explicit delete purpose |
| `provider.images()` | Generate images | Native text-to-image and subject-reference image-to-image request forms, model-specific validation, URL or base64 response representation |
| `provider.video()` | Create, query, list, cancel, or delete H3 V2 tasks | Multimodal H3 V2 content, 4–15 second duration, 768P/2K output, explicit open task states, no hidden polling |
| `provider.music()` | Generate music | Typed lyrics, generated-lyrics, instrumental, and cover request forms; non-streaming URL or decoded audio output |
| `provider.speech()` | Buffered synchronous synthesis, asynchronous submit, asynchronous query | Typed voice/audio settings, direct text or uploaded-text input for async tasks, explicit task state, no hidden polling |

These types are exported from `siumai::providers::minimax::resources` through the facade and from
`siumai_provider_minimax::resources` in the owning crate.

## Deliberate streaming limitations

Language generation supports established streams through `LanguageModel::stream`. Native resource
streaming is narrower:

- music requests are sent with streaming disabled and return one buffered response;
- synchronous speech uses the HTTP API with streaming disabled; the provider does not currently
  expose the MiniMax WebSocket speech stream;
- file downloads return bounded bytes rather than an unbounded byte stream;
- asynchronous speech and video expose submit/query operations and never start a hidden polling
  loop.

These omissions are intentional until Siumai has bounded, typed stream contracts that preserve the
provider lifecycle and terminal semantics. They must not be represented as generic streaming
support.

## Registry and host control plane

Registry is optional. It performs deterministic local lookup over a caller-configured registration:

```rust,no_run
use siumai::providers::minimax::{MinimaxCredential, MinimaxProvider};
use siumai::registry::{Registry, RegistryBuilderExt};

let provider = MinimaxProvider::builder(MinimaxCredential::api_key("test-key")).build()?;
let mut builder = Registry::builder();
builder.register_provider("minimax-primary", &provider)?;
let registry = builder.build()?;
let model = registry.language_model("minimax-primary:MiniMax-M3")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The host application remains responsible for account selection, aliases, default models, current
commercial availability, pricing, quota, compliance, health, weights, fallback, and deployment or
region policy. A custom endpoint accepted by the provider does not turn Siumai into an availability
control plane.

## Evidence

All named support in this document was checked against official MiniMax documentation on
2026-08-06. The local Vercel AI SDK checkout is secondary design and fixture evidence only.

| Scope | Official source |
|---|---|
| Anthropic-compatible Messages | [Messages API](https://platform.minimax.io/docs/api-reference/text-chat-anthropic) |
| Messages prompt caching | [Explicit Prompt Caching](https://platform.minimax.io/docs/api-reference/anthropic-api-compatible-cache) |
| OpenAI-compatible Chat Completions | [Chat Completions API](https://platform.minimax.io/docs/api-reference/text-chat-openai) |
| OpenAI-compatible Responses | [Create Response](https://platform.minimax.io/docs/api-reference/responses-create) |
| File lifecycle | [Upload](https://platform.minimax.io/docs/api-reference/file-management-upload), [List](https://platform.minimax.io/docs/api-reference/file-management-list), [Retrieve](https://platform.minimax.io/docs/api-reference/file-management-retrieve), [Download](https://platform.minimax.io/docs/api-reference/file-management-retrieve-content), [Delete](https://platform.minimax.io/docs/api-reference/file-management-delete) |
| Image generation | [Text to Image](https://platform.minimax.io/docs/api-reference/image-generation-t2i), [Image to Image](https://platform.minimax.io/docs/api-reference/image-generation-i2i) |
| H3 V2 video tasks | [Create](https://platform.minimax.io/docs/api-reference/video-generation-v2-create), [Query](https://platform.minimax.io/docs/api-reference/video-generation-v2-query), [List](https://platform.minimax.io/docs/api-reference/video-generation-v2-list), [Cancel or Delete](https://platform.minimax.io/docs/api-reference/video-generation-v2-delete) |
| Music generation | [Music Generation](https://platform.minimax.io/docs/api-reference/music-generation) |
| Speech synthesis | [HTTP T2A](https://platform.minimax.io/docs/api-reference/speech-t2a-http), [Create Async Task](https://platform.minimax.io/docs/api-reference/speech-t2a-async-create), [Query Async Task](https://platform.minimax.io/docs/api-reference/speech-t2a-async-query) |

Offline provider tests cover identity and authentication, API-mode routing, open future model IDs,
typed option validation, prompt-cache projection, request wire shapes, response decoding, task
states, bounds, and sanitized diagnostics. They do not perform live, credentialed, or billable
provider calls.
