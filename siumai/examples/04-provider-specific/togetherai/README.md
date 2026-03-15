# TogetherAI Examples

This directory contains TogetherAI-specific examples for the focused provider package surface.

## Package tier

- focused provider package
- current public focus: rerank
- distinct from the OpenAI-compatible `together` preset story

## Recommended path

1. use registry-first for app-level routing when possible
2. use config-first `siumai::provider_ext::togetherai::{TogetherAiConfig, TogetherAiClient}` for provider-specific setup
3. keep builders as convenience only

## Examples

- `rerank.rs` - config-first TogetherAI rerank with typed request helpers

## Notes

Do not merge this focused native provider story with the OpenAI-compatible `together` preset story. They have different package boundaries in Siumai.
