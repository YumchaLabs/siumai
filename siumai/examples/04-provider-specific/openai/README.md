# OpenAI Examples

This directory contains OpenAI-specific examples for the full provider-owned package surface.

## Package tier

- provider-owned package
- full extension surface with resources, streaming helpers, hosted tools, and provider-specific request options

## Recommended path

1. use registry-first for application architecture
2. use config-first `siumai::provider_ext::openai::{OpenAiConfig, OpenAiClient}` when you need OpenAI-specific setup
3. treat builder examples as convenience only

## Examples

- `computer_use.rs` - provider-hosted tool / computer-use workflow
- `file_search.rs` - OpenAI file search resource usage
- `file_search_results.rs` - file search results handling
- `responses-api.rs` - Responses API usage
- `responses-ext.rs` - provider-owned response extensions
- `responses-multi-turn.rs` - multi-turn Responses API flow
- `responses-streaming-tools.rs` - streaming + tools on Responses API
- `responses-websocket-incremental.rs` - websocket incremental workflow
- `stt_sse_streaming.rs` - transcription SSE streaming
- `tts_sse_streaming.rs` - speech SSE streaming
- `web_search.rs` - hosted web search flow

## Notes

OpenAI is the reference full provider-owned package in Siumai. This directory should remain strongly config-first and provider-extension oriented.
