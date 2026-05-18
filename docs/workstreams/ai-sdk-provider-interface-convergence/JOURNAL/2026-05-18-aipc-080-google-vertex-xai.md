# AIPC-080 Google Vertex xAI Slice

Date: 2026-05-18
Task: AIPC-080
Status: Done

## Summary

This slice split `@ai-sdk/google-vertex/xai` out as a first-class Siumai provider boundary instead
of folding it into native xAI or generic Vertex MaaS.

## Landed

- Added `google-vertex-xai` provider metadata, aliases, and registry factory wiring.
- Exposed `provider_ext::google_vertex_xai` plus `Provider::google_vertex_xai()` / `vertex_xai()`.
- Added provider-owned model/config exports and public-surface parity guards.
- Added request-body normalization so Google Vertex xAI strips shared reasoning knobs and stays
  aligned with the AI SDK package boundary.

## Validation

- Provider-focused no-network tests passed.
- Registry catalog/factory tests passed.
- Public import and public-path parity tests passed.
- Spec provider type mapping and unified-interface provider-type consistency checks passed.

## Boundary Note

Google Vertex root aliases remain aligned separately. Google/Gemini Interactions stays a distinct
deferred runtime lane and was not folded into this provider boundary.
