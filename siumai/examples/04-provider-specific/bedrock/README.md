# Amazon Bedrock Examples

This directory contains Bedrock-specific examples for the provider-owned surface.

## Available examples

- `chat.rs` - config-first Bedrock chat with typed request options
- `rerank.rs` - config-first Bedrock rerank with split agent-runtime ownership and typed request options

## Auth model

Siumai intentionally does **not** implement AWS SigV4 signing for Bedrock internally.

Recommended production patterns:

- inject fully signed SigV4 headers through `HttpConfig.headers`
- or send traffic through your own signed proxy / gateway

Compatibility/testing pattern:

- use `BEDROCK_API_KEY` as a Bearer token when your gateway accepts it

## Run

```bash
cargo run --example bedrock-chat --features bedrock
cargo run --example bedrock-rerank --features bedrock
```
