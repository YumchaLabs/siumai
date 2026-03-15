# Cohere Examples

This directory contains Cohere-specific examples for the focused provider package surface.

## Package tier

- focused provider package
- current focus: rerank
- do not expand this directory into a fake full-spectrum package without real provider-owned semantics

## Recommended path

1. use registry-first for application-level model selection when possible
2. use config-first `siumai::provider_ext::cohere::{CohereConfig, CohereClient}` for Cohere-specific setup
3. keep builders as convenience only

## Examples

- `rerank.rs` - config-first Cohere rerank with typed request helpers

## Notes

Cohere is intentionally narrow in Siumai today. Keep this directory focused on the capabilities Cohere actually owns in our public story.
