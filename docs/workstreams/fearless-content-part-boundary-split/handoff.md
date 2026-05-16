# Fearless ContentPart Boundary Split - Handoff

Last updated: 2026-05-16

## Current State

`CPB-020`, `CPB-030`, `CPB-040`, and `CPB-050` are complete. The refreshed direct `ContentPart` scan is recorded in
`direct-content-part-scan.md`, and the facade audit guard no longer auto-allows broad path buckets
such as `/provider_ext/`, `/mod.rs`, `/builder.rs`, `/config.rs`, or `/tests.rs`.

`CPB-030` migrated the Vertex Gemini image prompt path from direct `ContentPart::text(...)`
construction to `request_text_part(...)` and added behavior/source coverage.

`CPB-040` migrated the core streaming processor terminal text fallback from direct
`ContentPart::text(...)` construction to `response_text_part(...)` and added behavior/source
coverage.

`CPB-050` recorded ADR-0008: legacy `ContentPart` should not move immediately because it remains a
stable serde and `ChatMessage` / `ChatResponse` compatibility carrier. It is now classified as a
compatibility-only content surface, and a later breaking slice may move it under an explicit compat
namespace after directional adapters cover more main request and response paths.

This workstream is closed. Continue with a new workstream if the next slice starts moving additional
provider/protocol request serializers or response parsers off direct `ContentPart` construction.

## Continuation Notes

- Prefer adapter-first migrations over introducing a new broad public enum too early.
- Keep legacy `ContentPart` available until migration docs and fixture parity prove replacement
  paths.
- Use ADR-0008 as the decision anchor for any future `ContentPart` namespace move.
