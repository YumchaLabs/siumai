# MiniMax Provider

The MiniMax package owns the canonical `minimax` provider identity and its text, speech, image,
video, music, and file APIs. It uses provider-owned options on top of Siumai's stable request and
response types.

## Text Protocols

One client selects one explicit text protocol:

| `MinimaxChatEndpoint` | Route |
| --- | --- |
| `AnthropicMessages` (default) | `/anthropic/v1/messages` |
| `OpenAiChatCompletions` | `/v1/chat/completions` |
| `NativeText` | `/v1/text/chatcompletion_v2` |

All routes use `Authorization: Bearer <MINIMAX_API_KEY>`. The default API root is
`https://api.minimax.io`.

```rust
use siumai::prelude::unified::*;
use siumai::provider_ext::minimax::{
    MinimaxClient, MinimaxConfig, MinimaxOptions, MinimaxServiceTier, MinimaxThinking,
    MinimaxChatRequestExt,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let client = MinimaxClient::from_config(
    MinimaxConfig::new(std::env::var("MINIMAX_API_KEY")?)
        .with_model("MiniMax-M3")
        .with_thinking(MinimaxThinking::Adaptive)
        .with_service_tier(MinimaxServiceTier::Priority),
)?;

let request = ChatRequest::new(vec![ChatMessage::user("Hello").build()])
    .with_minimax_options(
        MinimaxOptions::new().with_thinking(MinimaxThinking::Adaptive),
    );
let response = client.chat_request(request).await?;
println!("{}", response.content_text().unwrap_or_default());
# Ok(())
# }
```

Current curated text profiles include M3 and the M2.7, M2.5, M2.1, and M2 families. Unknown model
IDs remain callable but receive no inferred capabilities. M3 supports adaptive or disabled
thinking. M2-family models always emit thinking, so Siumai rejects attempts to disable it instead
of silently accepting an ignored option.

MiniMax does not currently document a stable JSON object or JSON schema response contract for
these models. Stable `ResponseFormat` requests are therefore rejected rather than translated into
an invented wire field.

## Current Media Defaults

| Family | Default | Other curated models |
| --- | --- | --- |
| Speech | `speech-2.8-hd` | `speech-2.8-turbo`, Speech 2.6 and Speech 02 variants |
| Video | `MiniMax-Hailuo-2.3` | `MiniMax-Hailuo-2.3-Fast`, `MiniMax-Hailuo-02`, T2V-01 variants |
| Music | `music-2.6` | `music-2.6-free`, `music-cover`, `music-cover-free` |
| Image | `image-01` | `image-01-live` |

Music, video, TTS, and file operations expose provider-owned typed resources or extension traits
where their lifecycle cannot be represented honestly by the stable language-model interface.

## Configuration

```rust
use siumai::provider_ext::minimax::{MinimaxConfig, model_sets::MinimaxChatEndpoint};

let config = MinimaxConfig::new("api-key")
    .with_base_url("https://api.minimax.io")
    .with_model("MiniMax-M3")
    .with_chat_endpoint(MinimaxChatEndpoint::OpenAiChatCompletions);
```

Use an API-root custom URL. The provider normalizes legacy suffixes such as `/anthropic/v1` and
then appends the selected route exactly once.
