# Siumai Examples

Welcome to the Siumai examples directory.

Examples are organized by complexity and by the kind of public surface they demonstrate.
The recommended learning order is:

1. `registry-first`
2. `config-first`
3. `builder convenience`

## Directory Structure

```text
examples/
  01-quickstart/          # Start here
  02-core-api/            # Stable family APIs
  03-advanced-features/   # Middleware, retry, request shaping, advanced options
  04-provider-specific/   # Provider-owned, focused, and compat-vendor examples
  05-integrations/        # Registry, MCP, telemetry
  06-extensibility/       # Custom providers and executor-level extensibility
  07-applications/        # End-to-end application examples
```

## Construction Priority

When choosing examples for new code, prefer them in this order:

1. `registry-first` examples for application code and dynamic routing
2. `config-first` examples for provider-specific setup
3. `builder convenience` examples only for migration, comparison, or quick demos

Example map:

- registry-first: `01-quickstart/*`, `05-integrations/registry/*`
- config-first: most files under `04-provider-specific/*`
- builder convenience: explicit compatibility demos such as `04-provider-specific/openai-compatible/moonshot-siumai-builder.rs`

Surface policy:

- keep application walkthroughs on registry-first paths whenever possible
- keep the six Stable family modules `text`, `embedding`, `image`, `rerank`, `speech`, and `transcription` as the primary documented entry points
- keep provider-specific knobs near config-first examples and `provider_ext::<provider>` types
- keep builder examples explicit about being convenience or compatibility-oriented, not the default architecture

Provider example policy:

- examples under `04-provider-specific/<provider>` should prefer config-first when the provider owns a real package boundary
- focused provider directories should stay narrow instead of imitating the largest providers
- examples under `04-provider-specific/openai-compatible` are vendor-view or preset examples on top of the shared compat runtime, not evidence that every OpenAI-compatible vendor needs a dedicated provider package
- see `04-provider-specific/README.md` for the package-tier map across provider-specific examples
- builder-based files in provider directories should be labeled as convenience demos, not the preferred default path
- when looking for secondary-provider examples, start with config-first provider-owned files such as `04-provider-specific/deepseek/reasoning.rs`, `04-provider-specific/groq/structured-output.rs`, `04-provider-specific/xai/reasoning.rs`, `04-provider-specific/xai/structured-output.rs`, `04-provider-specific/xai/tts.rs`, `04-provider-specific/xai/web-search.rs`, `04-provider-specific/ollama/structured-output.rs`, and `04-provider-specific/ollama/metadata.rs`, then move to compat vendor views such as `04-provider-specific/openai-compatible/openrouter-embedding.rs`, `04-provider-specific/openai-compatible/openrouter-transforms.rs`, and `04-provider-specific/openai-compatible/perplexity-search.rs`, and only then to compat preset stories such as `04-provider-specific/openai-compatible/moonshot-basic.rs`, `04-provider-specific/openai-compatible/siliconflow-rerank.rs`, `04-provider-specific/openai-compatible/jina-rerank.rs`, `04-provider-specific/openai-compatible/voyageai-rerank.rs`, `04-provider-specific/openai-compatible/siliconflow-image.rs`, `04-provider-specific/openai-compatible/siliconflow-speech.rs`, `04-provider-specific/openai-compatible/siliconflow-transcription.rs`, `04-provider-specific/openai-compatible/together-image.rs`, `04-provider-specific/openai-compatible/together-speech.rs`, `04-provider-specific/openai-compatible/together-transcription.rs`, or `04-provider-specific/openai-compatible/fireworks-transcription.rs`

## Quick Start

### 1. Basic Chat

```bash
cargo run --example basic-chat --features openai
```

### 2. Streaming

```bash
cargo run --example streaming --features openai
```

### 3. Provider Switching

```bash
cargo run --example provider-switching --features "openai,anthropic,google"
```

## Learning Paths

### Beginner Path

1. `01-quickstart/basic-chat.rs` - simplest possible usage
2. `01-quickstart/streaming.rs` - real-time responses
3. `02-core-api/chat/chat-request.rs` - full-control `ChatRequest` + `text::generate` (recommended)
4. `02-core-api/chat/simple-chat.rs` - minimal `ChatRequest` + `text::generate`
5. `02-core-api/tools/function-calling.rs` - adding tools

### Developer Path

1. `02-core-api/chat/chat-request.rs` - recommended core API
2. `03-advanced-features/provider-params/structured-output.rs` - JSON schemas
3. `03-advanced-features/request-building/complex-request.rs` - full request control
4. `03-advanced-features/retry/retry-config.rs` - retry and resilience
5. `05-integrations/registry/basic-registry.rs` - string-driven model routing
6. `07-applications/chatbot.rs` - practical application example

### Advanced Path

1. `05-integrations/registry/custom-middleware.rs` - registry-first request/response middleware
2. `03-advanced-features/middleware/http-interceptor.rs` - config-first HTTP observability
3. `05-integrations/telemetry/basic-telemetry.rs` - registry-first telemetry attachment
4. `04-provider-specific/anthropic/prompt-caching.rs` - provider-owned optimization
5. `05-integrations/mcp/stdio-client.rs` - MCP integration
6. `06-extensibility/executor-testing.rs` - low-level executor validation

## Directory Guide

### `01-quickstart/`

Use this directory to learn the preferred Stable story quickly.

- `basic-chat.rs`
- `streaming.rs`
- `provider-switching.rs`

### `02-core-api/`

Use this directory to learn Stable family APIs such as:

- `text::generate`
- `text::stream`
- request building
- tools and tool loops
- multimodal requests

### `03-advanced-features/`

Use this directory for:

- structured output
- provider params
- middleware
- retry
- complex request construction

For observability and middleware, prefer runnable examples that stay on
registry-first or config-first paths. Treat `middleware_builder.rs` as a
middleware-composition utility demo, not as the default public construction
pattern for new code.

### `04-provider-specific/`

Use this directory when you need provider-specific setup or typed extension APIs.

See `04-provider-specific/README.md` for the provider package tier map.

### `05-integrations/`

Use this directory for:

- registry examples
- MCP integration
- telemetry

This directory is also the preferred home for application-level middleware
stories when registry ownership is part of the example.

### `06-extensibility/`

Use this directory when you want to build custom providers or work close to executor/runtime internals.

### `07-applications/`

Use this directory for practical end-to-end examples.

## Key Concepts

- Stable family API: `text`, `embedding`, `image`, `rerank`, `speech`, `transcription`
- provider-specific escape hatches: `provider_ext::<provider>`
- compatibility surface: thin, useful, but not the architectural default

## Running Examples

Most examples can be run with:

```bash
cargo run --example <example-name> --features <feature-list>
```

Examples under provider-specific directories often require provider credentials or local runtime setup.
Always read the file header and directory README first.

## Additional Resources

- `docs/workstreams/fearless-refactor-v4/public-api-story.md`
- `docs/workstreams/fearless-refactor-v4/migration-examples.md`
- `docs/workstreams/fearless-refactor-v4/provider-package-alignment.md`

## Contributing

When adding a new example:

- prefer registry-first for Stable application stories
- prefer config-first for provider-owned examples
- label builder examples as convenience demos
- place compat vendor examples under the compat narrative unless promotion criteria are met

Happy coding with Siumai.
