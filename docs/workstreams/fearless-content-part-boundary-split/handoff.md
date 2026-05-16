# Fearless ContentPart Boundary Split - Handoff

Last updated: 2026-05-16

## Current State

`CPB-020`, `CPB-030`, and `CPB-040` are complete. The refreshed direct `ContentPart` scan is recorded in
`direct-content-part-scan.md`, and the facade audit guard no longer auto-allows broad path buckets
such as `/provider_ext/`, `/mod.rs`, `/builder.rs`, `/config.rs`, or `/tests.rs`.

`CPB-030` migrated the Vertex Gemini image prompt path from direct `ContentPart::text(...)`
construction to `request_text_part(...)` and added behavior/source coverage.

`CPB-040` migrated the core streaming processor terminal text fallback from direct
`ContentPart::text(...)` construction to `response_text_part(...)` and added behavior/source
coverage.

The next executable task is `CPB-050`: record the compatibility decision for whether legacy
`ContentPart` can move under an explicit compatibility namespace in a later breaking slice.

## Continuation Notes

- Prefer adapter-first migrations over introducing a new broad public enum too early.
- Keep legacy `ContentPart` available until migration docs and fixture parity prove replacement
  paths.
- Run focused tests before updating milestone status.
