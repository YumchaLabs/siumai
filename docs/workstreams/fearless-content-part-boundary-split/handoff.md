# Fearless ContentPart Boundary Split - Handoff

Last updated: 2026-05-16

## Current State

`CPB-020` and `CPB-030` are complete. The refreshed direct `ContentPart` scan is recorded in
`direct-content-part-scan.md`, and the facade audit guard no longer auto-allows broad path buckets
such as `/provider_ext/`, `/mod.rs`, `/builder.rs`, `/config.rs`, or `/tests.rs`.

`CPB-030` migrated the Vertex Gemini image prompt path from direct `ContentPart::text(...)`
construction to `request_text_part(...)` and added behavior/source coverage.

The next executable task is `CPB-040`: pick one low-risk response projection path that still emits
legacy `ContentPart` directly and route it through a named response-side adapter.

## Continuation Notes

- Prefer adapter-first migrations over introducing a new broad public enum too early.
- Keep legacy `ContentPart` available until migration docs and fixture parity prove replacement
  paths.
- Run focused tests before updating milestone status.
