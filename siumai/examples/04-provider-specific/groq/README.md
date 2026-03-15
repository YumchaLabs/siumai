# Groq Examples

This directory contains Groq-specific examples for the provider-owned wrapper package surface.

## Package tier

- provider-owned wrapper package
- typed options and typed metadata belong to the Groq public surface

## Recommended path

1. use registry-first for application code
2. use config-first `siumai::provider_ext::groq::{GroqConfig, GroqClient}` for provider-specific setup
3. use builders only as compatibility convenience

## Examples

- `logprobs.rs` - Groq typed metadata / logprobs usage
- `structured-output.rs` - Groq structured-output usage on the provider-owned path

## Notes

Groq is not just a compat preset in the public story; it is a provider-owned wrapper surface with typed extension points.
