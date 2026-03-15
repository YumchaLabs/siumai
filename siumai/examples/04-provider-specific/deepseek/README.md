# DeepSeek Examples

This directory contains DeepSeek-specific examples for the provider-owned wrapper package surface.

## Package tier

- provider-owned wrapper package
- public construction stays provider-owned even when execution can reuse shared compat internals

## Recommended path

1. use registry-first for app architecture
2. use config-first `siumai::provider_ext::deepseek::{DeepSeekConfig, DeepSeekClient}` for provider-specific setup
3. treat builders as convenience only

## Examples

- `reasoning.rs` - DeepSeek reasoning-oriented usage with typed provider options

## Notes

DeepSeek should be documented as a provider-owned wrapper surface, not as just another generic OpenAI-compatible preset.
