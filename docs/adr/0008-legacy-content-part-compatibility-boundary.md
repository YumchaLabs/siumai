# ADR 0008: Legacy ContentPart compatibility boundary

- Status: accepted
- Date: 2026-05-16

## Context

`ContentPart` is a stable legacy carrier used by `ChatMessage`, `ChatResponse`, serde payloads, and
older provider/protocol glue. It currently carries both request-side `providerOptions` and
response-side `providerMetadata` on the same public enum variants.

That shape is backward-compatible, but it is not a strong architectural boundary. Request
serializers can accidentally learn from response metadata, and response parsers can accidentally
emit request provider options. The refactor has already introduced clearer directional surfaces:

- request prompt data uses `UserContentPart`, `AssistantContentPart`, `ToolContentPart`, and AI SDK
  V4 prompt parts.
- generated text output uses `GenerateTextContentPart`, output part carriers, and AI SDK V4 content
  parts.
- provider/protocol call sites that still need legacy payloads can use named request or response
  adapters.
- source guards now require direct `ContentPart` construction and provider-map usage to be audited.

## Options considered

### Option A - Move `ContentPart` to an explicit compatibility namespace immediately

Pros:

- Strongest public signal that the type is not the canonical request or response model.
- Reduces the chance that new examples continue to teach the legacy shape.

Cons:

- Breaks stable serde-facing `ChatMessage` and `ChatResponse` code before enough replacement paths
  are proven.
- Forces downstream migration while provider and protocol internals still legitimately use legacy
  payloads for parity.
- Turns an architectural cleanup into a broad public API break before the call-site migration is
  deep enough.

### Option B - Keep `ContentPart` in place as a canonical content model

Pros:

- Lowest migration cost.
- Preserves existing examples and user code unchanged.

Cons:

- Keeps request and response provider maps coupled.
- Makes future provider features likely to land on the broad legacy enum first.
- Conflicts with the Vercel-aligned split between prompt input, generated output, and compatibility
  edges.

### Option C - Keep `ContentPart` in place for now, but classify it as compatibility-only

Pros:

- Preserves stable payloads while the directional surfaces mature.
- Gives contributors a clear rule for new code without forcing an immediate breaking move.
- Allows a later breaking slice to move or re-export the type once migration docs, adapters, and
  fixture parity are complete.

Cons:

- Some compatibility usage remains visible in public imports during the transition.
- Requires source guards and documentation discipline until the later breaking slice lands.

## Decision

Choose Option C.

Legacy `ContentPart` remains available in its current public paths during the current beta line, but
it is classified as a compatibility carrier rather than the canonical request or response content
interface.

New architecture work should follow these rules:

1. Request-side construction should prefer `ModelMessage`, `UserContentPart`,
   `AssistantContentPart`, `ToolContentPart`, and AI SDK V4 prompt parts.
2. Response-side projection should prefer `GenerateTextContentPart`, output part carriers, stream
   output parts, and AI SDK V4 content parts.
3. Direct `ContentPart::...` construction in production code is only acceptable inside audited
   compatibility adapters or narrowly documented legacy payload paths.
4. Request adapters must not read response `providerMetadata`.
5. Response adapters must not emit non-empty request `providerOptions` unless preserving an
   explicitly documented legacy compatibility payload.

## Future breaking slice

Moving `ContentPart` under an explicit compatibility namespace is deferred until these conditions
are met:

1. The main request serializers are routed through directional request adapters or canonical prompt
   content types.
2. The main response parsers/projections are routed through directional response adapters or
   generated output content types.
3. Public docs and migration examples recommend request/response-specific imports instead of
   `ContentPart` for new code.
4. Source guards still report no unclassified high-value production direct construction sites.
5. Fixture parity proves that serde and provider protocol payloads do not regress.

The later breaking slice may either move the type to `compat` directly or keep a deprecated re-export
from the old path for one release cycle. That choice should be made based on downstream migration
cost at the time of the breaking release.

## Consequences

- `ContentPart` stays usable for existing code and stable serialized payloads.
- New code has an explicit direction: prompt parts for input, output/content parts for generated
  responses, compatibility adapters at the edge.
- Architecture tests and audits become the enforcement mechanism until the breaking move is safe.
- Migration documentation can teach the target shape now without forcing an immediate source break.
