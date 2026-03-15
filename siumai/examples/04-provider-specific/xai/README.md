# xAI Examples

This directory contains xAI-specific examples for the provider-owned wrapper package surface.

## Package tier

- provider-owned wrapper package
- provider-specific request options and metadata stay under `provider_ext::xai`

## Recommended path

1. use registry-first for application architecture
2. use config-first `siumai::provider_ext::xai::{XaiConfig, XaiClient}` for xAI-specific setup
3. keep builders as convenience only

## Examples

- `grok.rs` - xAI chat usage on the provider-owned wrapper path
- `reasoning.rs` - xAI reasoning defaults plus `ChatResponse::reasoning()` on the config-first path
- `structured-output.rs` - Stable `response_format` plus typed JSON extraction on the config-first xAI path
- `web-search.rs` - xAI web-search options and typed metadata usage
- `tts.rs` - xAI provider-owned text-to-speech path

## Notes

xAI should remain a provider-owned wrapper in the public story. Its speech/TTS path is also provider-owned rather than part of the shared compat audio family.
