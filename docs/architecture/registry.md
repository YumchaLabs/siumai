# Registry Contract

- Status: Current repository contract
- Updated: 2026-08-10
- Owner: `siumai-registry`
- Related decision: `docs/adr/0013-provider-identity-and-family-registration.md`

## Purpose

Registry is an immutable, network-free map from a host-owned route ID to one host-selected provider
registration. It provides deterministic local model resolution while keeping provider construction,
remote discovery, and business routing outside the lookup path.

A provider registration contains one or more default family bindings for one canonical provider.
Every binding carries its own exact technical scope and narrow constructor. This lets one route
expose, for example, compatible language and native transcription without falsely assigning one
protocol to the whole provider or making mutable model advice part of execution.

A registration contains at most one binding per family. Providers with multiple API modes for the
same family expose mode-specific registrations; the facade's `ProviderRegistrationSource` selects
only the provider's documented ergonomic default, and the host may register alternate modes under
separate route IDs.

The facade adapter is fallible because a provider may be validly configured only for native
resources or jobs. In that case it returns `NoPortableFamilyRegistration`; Registry never stores an
empty registration. A combined registration may be narrowed with `for_family` before registration.
Merging disjoint bindings with the same canonical provider ID is explicit host composition and does
not prove shared credentials, endpoint, account, or runtime origin.

Registry resolves the requested family directly from the selected registration. It has no model
policy callback, support-state evaluation, lifecycle warning injection, or advisory query.

## Route semantics

`RouteId` is an opaque local key such as `primary`, `fast`, or `tenant-a-eu`. Registry does not infer
commercial provider identity, geography, pricing, availability, or fallback from that text. The host
may attach any business meaning it wants before building the immutable Registry snapshot.

Model references combine a route and an open model ID. Resolving a model is synchronous and does not
contact a provider. Unknown, rolling, deprecated, and retired model IDs remain constructible. A
concrete provider may still reject a stable technical constraint when planning the request, and the
remote service remains authoritative for mutable product capability.

## Host-owned policy

Applications that require model allowlists, lifecycle warnings, commercial availability checks, or
compliance policy evaluate those concerns before Registry resolution. Keep the configured provider
or its support manifest beside the route definition, inspect the dated evidence explicitly, then
decide whether to expose or call the route. Registry intentionally does not provide a generic
replacement for the removed `Registry::evaluate` API.

This separation keeps support evidence useful without allowing stale catalogs to block a valid
future model or mutate explicit provider options. Rebuilding an immutable Registry snapshot remains
the host's mechanism for changing routes after its own policy changes.

## Non-goals

Registry does not:

- own credentials, endpoints, retry configuration, or provider builders;
- discover remote model/deployment inventories;
- cache model objects, TTLs, LRUs, or singleflight construction state;
- implement provider fallback, load balancing, quota, health, or cost policy;
- depend on built-in provider packages;
- infer support from provider profiles or support manifests;
- evaluate model lifecycle, allowlist, or product-capability policy;
- choose between API modes within one family binding;
- expose a universal provider factory or generic client.

Those concerns belong to configured providers or the host control plane. Applications that need
dynamic routing may rebuild and atomically replace a Registry snapshot, or place their own routing
layer above Registry.

## Dependency boundary

`siumai-registry` depends only on provider-neutral contracts. Provider packages construct their own
registrations; the `siumai` facade supplies feature-gated ergonomic adapter implementations. This
keeps custom/internal providers first-class and prevents Registry from becoming a central provider
switch.
