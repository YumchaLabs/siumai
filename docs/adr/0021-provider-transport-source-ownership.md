# ADR-0021: Provider Transport Source Ownership

## Status

Accepted

## Date

2026-09-02

## Context

The OpenAI-compatible and Anthropic-compatible engines each need to accept one
of two transport inputs: a provider-authorized `ProviderTransportBuilder` that
must be configured and consumed, or an already configured `ProviderTransport`
that must be reused without mutation. Before this decision, both engines
implemented the same settings-conflict check, transport construction, and
typed base-URL-plus-policy comparison locally.

That duplication made a security-sensitive ordering rule easy to drift. It also
made provider code appear to own transport mechanics, even though credentials,
provider identity, retry declarations, endpoint selection, and configuration
error mapping belong to the provider layer. A broad provider builder or
middleware abstraction would move those authorities in the wrong direction.

## Decision

Add the experimental direct-crate contract
`siumai_transport::experimental::provider_author::ProviderTransportSource`.

The source has exactly two internal states:

- a configured state owning one complete `ProviderTransportBuilder`;
- a prebuilt state owning one `ProviderTransport`.

The source is single-use and is consumed by `resolve(&EndpointConfig)`. A
configured source may receive repeated complete settings snapshots; the last
snapshot wins and is applied before the builder is consumed. A prebuilt source
records any settings override and reports the settings conflict at consuming
resolution, before endpoint comparison. Successful prebuilt resolution returns
the original cloneable transport runtime rather than rebuilding or
reauthenticating it.

Resolution compares the typed `EndpointConfig` base URL and `EndpointPolicy`
using the same equality semantics as the former compatibility-engine helpers.
The source does not compare raw strings, hosts, origins, model names, or
provider identity, and it does not perform DNS or network I/O.

The source error is an explicit provider-neutral classification with only:

1. settings conflict;
2. endpoint mismatch; and
3. the underlying `TransportConfigError` as its direct source.

Compatibility engines map the first two classifications to their existing
provider-specific errors and preserve the existing transparent transport
error source chain. The source accepts no credential, profile, provider error
mapper, resolver identity, retry classifier identity, or callback.

The module remains outside the transport root re-exports, facade, and preludes.
It is documented as experimental for direct transport users and provider
authors. Promotion requires a third independent real consumer that proves the
same configured-versus-prebuilt invariant; otherwise the module may be deleted
without creating a stable compatibility obligation.

## Options considered

### Option A: Keep both local helpers

Rejected. The logic is identical and security-sensitive, so local copies create
avoidable drift and make endpoint/settings ordering harder to audit.

### Option B: Add a universal provider builder or middleware layer

Rejected. Provider construction, credentials, and request policy would become
owned by a higher layer that cannot enforce them completely. No current
consumer needs a general middleware contract.

### Option C: Share only an endpoint-comparison function

Rejected for the first milestone. It would leave settings conflict and builder
consumption duplicated, preserving the main lifecycle seam's drift risk.

### Option D: Add the narrow experimental provider-author source

Chosen. It removes the repeated transport-resolution mechanics while keeping
provider-specific authorities and public error contracts in their current
owners.

## Consequences

### Positive

- Both compatibility engines use one audited settings and endpoint-resolution
  order.
- Prebuilt transports remain immutable, reusable, and identity-preserving.
- Provider crates retain credentials, profiles, replay declarations, retry
  policy, and public configuration errors.
- The experimental surface can be evaluated by real consumers without adding a
  facade or stable root API.

### Costs

- Provider authors must depend on an explicitly experimental module and map its
  closed errors.
- A third consumer is required before the seam can be considered for
  promotion.
- The source intentionally does not solve provider construction or general
  request middleware.

## Verification

Transport unit and integration contracts cover configured and prebuilt states,
last-write-wins settings, single-use consumption, prebuilt conflict priority,
typed URL/policy compatibility (including URL normalization and trailing
slashes), runtime identity preservation, transport error sources, and redacted
debug output. Compatibility-engine migration tests remain the authority for
provider error mapping and credential-validation order.

## References

- `docs/architecture/transport-contract.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `docs/plans/2026-09-02-1808-refactor-architecture-deepening-transport-assembly-plan.md`
