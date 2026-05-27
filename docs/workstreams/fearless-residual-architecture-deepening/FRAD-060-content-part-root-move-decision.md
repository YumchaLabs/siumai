# FRAD-060 ContentPart Root Move Decision

Status: DONE_WITH_CONCERNS
Date: 2026-05-27

## Decision

Do not remove or move the low-level `siumai-spec::types::ContentPart` and
`siumai-core::types::ContentPart` root paths in FRAD-060.

Instead, FRAD-060 moves high-value production usages that intentionally build or project the legacy
carrier to explicit compatibility imports:

- `siumai-spec::types::compat::content::*`
- `siumai-core::compat::content::*`
- `siumai::__private::compat::content::*` for facade macro expansion

This keeps the public serde-facing type identity stable while making new production code cross the
compatibility boundary deliberately.

## Why Not Move The Root Path Now

The root move is still blocked by the same ADR-0008 conditions:

- `ChatMessage` and `ChatResponse` still serialize legacy `MessageContent` / `ContentPart`
  payloads.
- `siumai-spec::types::*` and `siumai-core::types::*` still expose blanket root re-exports, so a
  partial move would create split identities rather than a clean compatibility namespace.
- Provider, protocol, and bridge response paths still preserve provider-native `ChatResponse` /
  `ContentPart` shapes for wire parity and provider metadata.
- The current gates prove serde parity and production compat imports, but not a full provider and
  protocol root-move fixture suite.

This matches the local Vercel AI SDK reference direction: prompt/model-message content and generated
output content are separate contracts, while Siumai's legacy `ContentPart` remains only a
compatibility carrier during the beta line.

## Evidence

- `cargo fmt --check -p siumai-spec -p siumai-core -p siumai`
- `cargo check -p siumai-core -p siumai-spec -p siumai`
- `cargo nextest run -p siumai-spec --no-default-features --test content_projection_boundary_test adr_0008 --no-fail-fast`
- `cargo nextest run -p siumai-spec content --no-fail-fast`
- `cargo nextest run -p siumai --test public_surface_imports_test --no-fail-fast`
- `cargo nextest run -p siumai --test facade_architecture_boundary_test content --no-fail-fast`
- `cargo nextest run -p siumai test_macros --no-fail-fast`

## Follow-On Gate

A future root move must add a full-root-move fixture suite that covers `ChatMessage`,
`ChatResponse`, provider/protocol response fixtures, bridge response fixtures, and migration docs in
one breaking slice.
