# Migration Guide: `0.11.0-beta.6` -> `0.11.0-beta.7`

This guide covers the user-visible migration points for `0.11.0-beta.7`.
Most applications that only call chat/text helpers can upgrade by changing the Cargo version.
The main source changes affect advanced users who implement model traits, inspect streaming events,
construct shared structs directly, or compare serialized snapshots.

## TL;DR

- Normal text/chat callers: usually no source migration.
- Model trait generics: use the canonical family traits (`TextModel`, `EmbeddingModel`,
  `ImageModel`, `RerankingModel`, `SpeechModel`, `TranscriptionModel`, `VideoModel`).
- Streaming consumers: prefer semantic accessors such as `event.text_delta()` and
  `event.part_ref()`.
- Usage and metadata snapshots: expect AI SDK-aligned field names and provider-rooted metadata.
- Raw provider envelopes: opt in through request/response retention controls instead of parsing
  debug output.

If you are upgrading from `0.11.0-beta.5` or earlier, read this guide after
`docs/migration/migration-0.11.0-beta.6.md`.

## 1) Model-family traits

`0.11.0-beta.7` finishes the trait-name cleanup started in the previous beta.
The stable Rust family names are now:

- `TextModel`
- `EmbeddingModel`
- `ImageModel`
- `RerankingModel`
- `SpeechModel`
- `TranscriptionModel`
- `VideoModel`

Use these names for application-level generics and helper functions.

```rust,ignore
use siumai::prelude::unified::{ChatRequest, TextModel, text};

async fn run_text<M: TextModel + ?Sized>(model: &M, request: ChatRequest) -> anyhow::Result<()> {
    let response = text::generate(model, request, text::GenerateOptions::default()).await?;
    println!("{}", response.text);
    Ok(())
}
```

If you specifically need a full language model handle with metadata, `LanguageModel` remains
available:

```rust,ignore
use siumai::prelude::unified::LanguageModel;

fn accepts_language_model<M: LanguageModel + ?Sized>(_model: &M) {}
```

`LanguageModelV4`, `ImageModelV4`, and `VideoModelV4` are provider-contract markers for
AI SDK-aligned low-level integrations. Most application code should not use them unless an
explicit helper requires them.

## 2) Streaming events

The canonical stream payload is now `ChatStreamPart`, carried by
`ChatStreamEvent::Part` / `ChatStreamEvent::PartWithReplay`. Application code should prefer
semantic accessors so it stays compatible as more AI SDK stream parts are exposed.

Before:

```rust,ignore
match event {
    ChatStreamEvent::TextDelta { delta, .. } => print!("{delta}"),
    _ => {}
}
```

After:

```rust,ignore
if let Some(delta) = event.text_delta() {
    print!("{delta}");
}
```

For richer stream consumers:

```rust,ignore
if let Some(delta) = event.reasoning_delta() {
    eprintln!("reasoning: {delta}");
}

if let Some(part) = event.part_ref() {
    // Inspect source, tool, metadata, finish, custom, file, and reasoning-file parts here.
}
```

## 3) Usage values

`Usage` now keeps AI SDK-style `inputTokens` / `outputTokens` as the canonical stable layer while
legacy prompt/completion/total counts remain available through accessors and compatibility serde.

Avoid struct literals or direct legacy field reads.

Before:

```rust,ignore
let prompt = usage.prompt_tokens;
let completion = usage.completion_tokens;
```

After:

```rust,ignore
let prompt = usage.prompt_tokens();
let completion = usage.completion_tokens();
let total = usage.total_tokens();

let input = &usage.input_tokens;
let output = &usage.output_tokens;
```

Construction should use constructors or the builder:

```rust,ignore
let usage = Usage::new(12, 7);

let usage = Usage::builder()
    .with_input_total_tokens(12)
    .with_output_text_tokens(7)
    .build();
```

## 4) Serialized snapshots and metadata

If your tests compare JSON snapshots, update them for the AI SDK-aligned shapes.
Common differences are:

- `ToolChoice` and `FinishReason` serialize with AI SDK public values.
- `Warning` now has explicit compatibility/deprecated/unsupported shapes.
- `Usage` prefers `inputTokens`, `outputTokens`, and `raw`.
- `ResponseMetadata` serializes `modelId` / `timestamp`.
- `providerMetadata` is provider-rooted (`provider id -> object`) across response and stream
  metadata.
- Stream parts may appear as stable `ChatStreamPart` values rather than provider-specific custom
  payloads.

## 5) Raw provider request/response envelopes

For audit, debugging, or gateway work, use the new request/response body retention controls exposed
by the AI SDK-style helper options. Do not parse formatted debug strings; inspect the structured
metadata and retained body values instead.

## 6) Provider-specific note: Vertex Gemini image

Vertex Gemini image requests now reject mask and multi-image count settings on the Gemini image
path when those settings are unsupported. This is intentional: it prevents a request from silently
being routed through the wrong provider mode.

If you need mask/reference-image behavior, use the Vertex Imagen edit path and its provider options
instead of the Gemini image generation path.

## 7) Live provider smoke tests

The live smoke script still skips missing API keys. In `0.11.0-beta.7`, Gemini defaults to
`gemini-2.5-flash-lite` for a more stable low-cost smoke path, and transient provider/network
errors are retried.

```powershell
$env:SIUMAI_TEST_PROXY = "http://127.0.0.1:10809"
$env:SIUMAI_ENV_SMOKE_PROFILE = "all-providers"
scripts\test-env-smoke.bat
```

Set `SIUMAI_ENV_SMOKE_STRICT=1` when CI should fail instead of self-skipping known account, region,
or quota denials.
