# Ollama Examples

This directory contains Ollama-specific examples for the provider-owned package surface.

## Package tier

- provider-owned package
- local-runtime oriented provider story

## Recommended path

1. use registry-first for application architecture when routing matters
2. use config-first `siumai::provider_ext::ollama::{OllamaConfig, OllamaClient}` for local provider setup
3. keep builders as convenience only

## Examples

- `local-models.rs` - local model usage and setup
- `structured-output.rs` - Stable `response_format` plus typed JSON extraction on the config-first Ollama path
- `metadata.rs` - Ollama typed metadata usage

## Notes

Ollama should stay provider-owned and local-runtime focused rather than being forced into a generic cloud-provider documentation shape.
