# ADR-0011: Keep Protocol Projection at Owning Boundaries

## Status

Accepted

## Date

2026-08-05

## Context

Siumai previously maintained a `siumai-bridge` package for converting requests, responses, and
streams among OpenAI, Anthropic, and Gemini representations. The package evolved around the legacy
`ChatRequest`, `ChatResponse`, and `ChatStreamEvent` contracts and accumulated its own protocol
target enum, loss policy, customization hooks, stream lifecycle adapters, and wire serializers.

The current architecture already has two stronger ownership seams:

- protocol packages map between canonical model-family contracts and their wire formats;
- concrete server or gateway integrations own the external wire contract they expose.

No workspace package consumes `siumai-bridge`. Preserving it would therefore keep a second codec and
middleware architecture without a current product flow. A minimal package containing only reports
and hooks would still be a shallow abstraction: it would have a large public surface, no driver,
and no consumer that can prove its loss semantics.

The local AI SDK reference uses provider-owned wire mapping and model middleware for request,
generation, and stream wrapping. It does not require a general package for arbitrary
provider-protocol transcoding.

## Decision

Remove `siumai-bridge` from the workspace and keep projection behavior at the boundary that owns it.

1. A protocol package owns canonical request encoding, response decoding, stream state, wire
   validation, and protocol-specific metadata projection.
2. A provider package owns provider defaults, API-mode selection, typed options, authentication,
   and provider-specific restrictions around those codecs.
3. A server or gateway integration depends directly on the protocol codec for the wire contract it
   exposes and reports unsupported or lossy behavior at that concrete boundary.
4. Provider-neutral request, response, and stream customization belongs to a language-model
   middleware seam, not to a protocol bridge.
5. Raw JSON, headers, credentials, endpoints, and transport policy are not exposed through generic
   bridge hooks. They remain protected by provider, protocol, or transport owners.
6. Siumai does not maintain a closed global enum of supported protocol targets.

A shared projection package may be introduced later only after multiple real consumers demonstrate
the same canonical-only behavior. Such a package must be strict by default, use open protocol
identities, return a bounded structured loss report, and avoid duplicating protocol stream state
machines.

## Options considered

### Option A: Keep repairing the legacy bridge

Rejected because it preserves legacy model contracts and requires canonical protocol codecs to
implement obsolete stream and serializer traits.

### Option B: Keep only bridge reports and customization hooks

Rejected because no runtime or integration consumes them. The hooks would duplicate model
middleware and typed protocol/provider options while providing no complete projection operation.

### Option C: Rewrite all cross-protocol conversions now

Rejected because there is no current consumer requiring direct protocol-to-protocol transcoding.
Implementing every direction would duplicate codec state machines and create a broad feature matrix
for speculative behavior.

### Option D: Remove the package and localize projection

Chosen because it follows existing ownership, reduces the public surface, and lets future
abstractions be extracted from proven integrations rather than predicted combinations.

## Consequences

### Positive

- Protocol behavior has one implementation owner.
- Server and gateway adapters expose only the protocols they actually support.
- Model middleware and wire customization no longer compete as parallel extension systems.
- Adding a provider does not require extending a global bridge target enum.
- Workspace builds no longer carry unused protocol combinations and duplicate fixtures.

### Costs

- Direct users of the unpublished next-version bridge surface must migrate or retain the previous
  release.
- A future product that genuinely needs arbitrary protocol transcoding must define and test that
  workflow before a shared package is extracted.
- Useful legacy round-trip fixtures should be recreated at the owning protocol or integration
  boundary when that behavior is required.

## Migration

1. Remove `siumai-bridge` from workspace membership and shared dependencies.
2. Remove its legacy contracts, codecs, customization hooks, examples, and CI lane.
3. Keep canonical wire encoding and decoding tests in protocol packages.
4. Let `siumai-server` add direct protocol dependencies only for externally exposed compatibility
   routes.
5. Design the canonical language-model middleware independently of wire formats.

## References

- `repo-ref/ai/packages/provider/src/language-model/v4/language-model-v4-middleware.ts`
- `repo-ref/ai/packages/ai/src/middleware/wrap-language-model.ts`
- `repo-ref/ai/packages/gateway/src/gateway-provider.ts`
- `docs/architecture/overview.md`
- `docs/architecture/public-api.md`
