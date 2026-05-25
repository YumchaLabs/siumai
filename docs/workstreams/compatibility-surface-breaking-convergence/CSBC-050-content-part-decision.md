# CSBC-050 ContentPart Breaking-Slice Decision

Status: DONE_WITH_CONCERNS
Date: 2026-05-25

## Decision

Do not move the low-level `siumai-spec::types::ContentPart` /
`siumai-core::types::ContentPart` root paths in this slice.

The safe facade-level break has already landed in the earlier
`content-part-compat-namespace-break` lane:

- `siumai::prelude::unified::*` no longer exports legacy `ContentPart`.
- Migration code has explicit paths:
  - `siumai::compat::content::*`
  - `siumai::content::compat::*`
  - `siumai::prelude::compat::content::*`
  - `siumai_core::compat::content::*`
- `siumai-spec/src/types/ai_sdk/response_compat_projection.rs` owns the
  response-side legacy `ContentPart` to generated-output projection.

CSBC-050 therefore records the remaining blockers for a full root namespace move
and adds source guards instead of forcing a broad break.

## Blockers

1. `ChatMessage` and `ChatResponse` remain serde-facing public payloads whose
   legacy shape is still `MessageContent` / `ContentPart`. Moving the root
   spec/core type identity now would create a broad low-level public break before
   replacement payload fixtures are proven.
2. `siumai-spec::types::*` still re-exports `chat::*`, and
   `siumai-core::types::*` is a blanket re-export of `siumai-spec::types::*`.
   Moving only one layer would create split type identities instead of a clean
   compatibility namespace.
3. Provider, protocol, and bridge response paths still parse or serialize
   provider-native `ChatResponse` / `ContentPart` shapes directly when preserving
   wire parity and provider metadata.
4. The shipped fixture coverage proves the facade/prelude break and the named
   response adapter seam, but it is not a dedicated full-root-move parity suite
   for every provider/protocol payload.

## Guard

`siumai-spec/tests/content_projection_boundary_test.rs` contains
`adr_0008_full_contentpart_namespace_break_blockers_are_guarded`, which checks:

- explicit spec/core/facade compatibility namespaces exist;
- root spec chat re-exports remain intentionally tied to serde-facing payloads;
- ADR-0008 names the current partial break and the remaining root-move blockers;
- this decision record keeps the blockers concrete.

## Follow-On

Open a separate root-namespace migration lane only after:

- `ChatMessage` / `ChatResponse` replacement or alias behavior is specified;
- provider/protocol response fixture parity covers the root move;
- downstream migration docs can give a single low-level replacement path without
  split type identity;
- ADR-0008 is updated from compatibility classification to an actual removal
  plan.
