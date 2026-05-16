# Fearless ContentPart Boundary Split - TODO

Last updated: 2026-05-16

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[d]` deferred with rationale

## CPB-010 - Workstream Framing

- [x] Create design, TODO, milestones, evidence, handoff, and machine-readable workstream docs.
- [x] Link the workstream from `docs/README.md`.

Validation:

- Documentation review and `git diff --check`.

## CPB-020 - Legacy ContentPart Guard Tightening

- [x] Refresh the production direct-construction scan from
      `fearless-spec-core-boundary-convergence/content-part-construction-audit.md`.
- [x] Add or tighten a guard that fails when new production direct `ContentPart::...`,
      `provider_options:`, or `provider_metadata:` hits appear outside audited adapter paths.
- [x] Record accepted adapter paths and false-positive buckets in this workstream.

Validation:

- `cargo nextest run -p siumai --test facade_architecture_boundary_test content_part_provider_map_audit_covers_high_value_production_hits --features openai,anthropic,google --no-default-features --no-fail-fast`
- `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`

Notes:

- The refreshed scan found 121 paths covered by the existing spec/core audit, 48 explicitly
  recorded low-priority or false-positive paths, and 0 unclassified high-value production paths.
- The facade guard now reads `direct-content-part-scan.md` instead of allowing broad
  `/provider_ext/`, `/mod.rs`, `/builder.rs`, `/config.rs`, and `/tests.rs` buckets.

## CPB-030 - Request Adapter Migration Slice

- [x] Pick one low-risk request construction path that still emits legacy `ContentPart` directly.
- [x] Route it through a named request-side adapter that guarantees no response metadata is read.
- [x] Add behavior or source coverage for that adapter.

Validation:

- Focused crate tests for the touched request path.
- `cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast`

Notes:

- `siumai-provider-google-vertex/src/standards/vertex_gemini_image.rs` now routes Gemini image
  prompt text through `request_text_part(...)` instead of direct `ContentPart::text(...)`.
- The existing centralized request-content guard now covers both text and file adapters.
- A behavior test verifies prompt text emits empty request provider options and no response
  provider metadata.

## CPB-040 - Response Adapter Migration Slice

- [x] Pick one low-risk response parser/projection path that still emits legacy `ContentPart`
      directly.
- [x] Route it through a named response-side adapter that guarantees request provider options are
      not emitted except for empty legacy defaults.
- [x] Add behavior or source coverage for that adapter.

Validation:

- Focused crate tests for the touched response path.
- Relevant provider/protocol fixture parity test.

Notes:

- `siumai-core/src/streaming/processor.rs` now routes terminal text fallback projection through
  `response_text_part(...)` instead of direct `ContentPart::text(...)`.
- Source coverage prevents `StreamProcessor` response consolidation from reintroducing
  `ContentPart::text(...)` direct calls.
- Behavior coverage verifies the adapter emits empty request provider options and no response
  provider metadata for plain terminal text.

## CPB-050 - Compatibility Surface Decision

- [x] Decide whether legacy `ContentPart` can move under an explicit compatibility namespace in the
      next breaking slice.
- [x] Update migration docs with canonical request/response content imports.
- [x] Close or split follow-up tasks based on the refreshed scan.

Validation:

- `cargo fmt` for touched crates.
- Focused `cargo nextest` runs for affected crates.
- `git diff --check`.

Notes:

- ADR-0008 records the decision: do not move legacy `ContentPart` immediately because it is still a
  stable serde and `ChatMessage` / `ChatResponse` compatibility carrier. Classify it as
  compatibility-only now, then revisit a breaking namespace move after more request and response
  adapters cover the main paths.
- `migration-0.11.0-beta.7.md` now points users at canonical request prompt parts, AI SDK V4 prompt
  parts, generated text output parts, and AI SDK V4 generated content parts.
- The refreshed scan has no unclassified high-value production paths, so there is no follow-up
  split inside this workstream. The remaining work is the future breaking namespace slice described
  by ADR-0008.
