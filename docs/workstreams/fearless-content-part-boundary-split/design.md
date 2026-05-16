# Fearless ContentPart Boundary Split - Design

Last updated: 2026-05-16

## Context

The spec/core boundary workstream closed with one intentionally deferred compatibility problem:
legacy `ContentPart` still carries request-side `providerOptions` and response-side
`providerMetadata` on the same enum variants.

That shape is useful for backward compatibility, but it is a shallow interface for new code. A
caller must know whether a `ContentPart` instance is being used as prompt input, generated output,
stream replay, UI data, or protocol adapter glue before it can safely interpret provider maps.
That weakens locality: request serializers can accidentally learn from response metadata, and
response parsers can accidentally retain request options.

The codebase already has the target direction:

- request-side non-V4 model messages live in `siumai-spec::types::prompt::{UserContentPart,
  AssistantContentPart, ToolContentPart}`.
- AI SDK V4 prompt/content projections are physically split under
  `siumai-spec::types::ai_sdk::language_model_v4`.
- response-side generated text projection lives in `GenerateTextContentPart`.
- direct legacy `ContentPart` construction is audited by
  `content-part-construction-audit.md` and guarded by facade/core/provider/protocol source tests.

## Decision

Treat legacy `ContentPart` as a compatibility carrier, not the canonical request or response
content interface.

New request construction should flow through prompt/model-message content parts. New response
projection should flow through generated output/content parts. Protocol, bridge, provider, and
facade code should use explicit adapter functions when they must cross into legacy `ContentPart`
for compatibility.

## Goals

- Make the request/response provider-map direction visible in public type names and adapter names.
- Reduce direct `ContentPart::...` construction in production code by replacing it with
  request-side or response-side adapters.
- Keep legacy `ContentPart` available during migration, but classify it under compatibility docs
  and guards.
- Preserve serde compatibility where legacy public payloads still require `ContentPart`.
- Add source guards so new production code cannot re-expand the legacy dual-map surface.

## Non-goals

- Do not remove every `ContentPart` variant in one patch.
- Do not break protocol fixture parity while provider-owned response parsers still emit legacy
  content for stable response structs.
- Do not add another broad public enum unless it replaces real call-site complexity.
- Do not change AI SDK V4 prompt/content shapes unless this work uncovers an actual mismatch.

## Target Shape

Allowed primary paths:

- request input: `UserContentPart`, `AssistantContentPart`, `ToolContentPart`, model-message
  projection helpers, and AI SDK V4 prompt parts.
- response output: `GenerateTextContentPart`, output-part carriers, source/file/tool output
  carriers, and AI SDK V4 generated content parts.
- compatibility edge: named adapters to/from legacy `ContentPart`.

Discouraged production paths:

- direct `ContentPart::...` construction outside adapter modules.
- request serializers reading `provider_metadata` or `providerMetadata`.
- response parsers emitting non-empty request-side `provider_options` except when explicitly
  preserving a legacy compatibility payload.

## First Slices

1. Add a dedicated compatibility note and source guard for new direct legacy `ContentPart`
   construction outside audited adapter paths.
2. Move one low-risk production request path from direct legacy `ContentPart` construction to a
   named request adapter.
3. Move one low-risk response projection path from direct legacy `ContentPart` construction to a
   named response adapter.
4. Re-run the audit and decide whether legacy `ContentPart` can be moved under an explicit
   compatibility namespace in a later breaking slice.

## Validation

Focused gates for this lane:

```text
cargo fmt --package siumai-spec --package siumai-core --package siumai-bridge --package siumai --check
cargo nextest run -p siumai-spec --test content_projection_boundary_test --no-default-features --no-fail-fast
cargo nextest run -p siumai-core --test core_provider_boundary_test --no-default-features --no-fail-fast
cargo nextest run -p siumai --test facade_architecture_boundary_test --features openai,anthropic,google --no-default-features --no-fail-fast
git diff --check
```
