# Anthropic Examples

This directory contains Anthropic-specific examples for the provider-owned package surface.

## Package tier

- provider-owned package
- prefer config-first construction through `siumai::provider_ext::anthropic`
- use builder flows only as convenience demos elsewhere

## Recommended path

1. use registry-first for application architecture
2. use config-first here when you need Anthropic-specific setup or typed extensions
3. keep raw provider options as a fallback, not the first choice

## Examples

- `extended-thinking.rs` - thinking-focused Anthropic usage
- `extended-thinking-ext.rs` - thinking through provider-owned extension helpers
- `prompt-caching.rs` - prompt caching workflow
- `thinking-replay-ext.rs` - replay / extension-oriented thinking flow
- `web-search.rs` - provider-hosted web search flow

## Notes

Anthropic belongs to the full provider-owned package tier, so this directory should continue to emphasize config-first usage and typed provider extensions.
