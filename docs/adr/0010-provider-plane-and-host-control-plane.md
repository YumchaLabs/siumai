# ADR-0010: Separate the Provider Plane from the Host Control Plane

## Status

Accepted

## Date

2026-08-05

## Context

Provider SDK code can reliably own authentication, endpoint construction, transport behavior,
protocol codecs, typed provider options, and provider resources. It cannot reliably own the mutable
state of a customer's cloud account or application deployment.

Region-specific model availability, deployment inventory, quota, pricing, rollout state, data
residency policy, fallback order, and account entitlements can differ between accounts and can change
without an SDK release. Encoding that state into Rust enums, support claims, or static model matrices
creates a false control plane: the types look authoritative even though the provider remains the real
source of truth.

The previous design direction allowed provider profiles to include account, region, deployment, model
availability, recommended defaults, and lifecycle matrices. The DashScope profile made the problem
concrete by combining endpoint construction with deployment scopes and region-by-model tables.

The local AI SDK reference demonstrates a smaller seam. Its Alibaba package exposes an Alibaba
provider with configurable endpoints and open model IDs. DashScope remains an implementation detail
of the endpoint and native wire protocols. Its generic Registry maps caller-configured providers to
model IDs without discovering cloud inventory or choosing a region.

## Decision

Siumai separates an execution-oriented **provider plane** from the application's **host control
plane**.

### Provider plane

A provider crate owns:

- canonical vendor/product identity;
- credential loading and application;
- endpoint parsing, construction, validation, and credential audience;
- HTTP, SSE, WebSocket, retry, cancellation, and resource safety policy;
- wire codecs and stream terminal semantics;
- typed provider options and metadata;
- request-time protocol and encoding validation;
- provider resources such as a remote catalog client when the provider actually exposes one;
- lightweight synchronous model construction for open model IDs.

A provider may accept host-selected technical coordinates when an API requires them. Examples include
an AWS signing region, a Vertex project/location, a workspace hostname component, an API version, or
a custom base URL. Such values are configuration inputs, not an SDK-owned inventory. The provider
validates and uses them but does not choose them or infer business availability from them.

### Host control plane

The application or a separate control-plane component owns:

- tenant, account, project, workspace, subscription, and deployment selection;
- region selection, data-residency and compliance policy;
- current model allowlists and account entitlements;
- default models, aliases, rollout policy, weights, health routing, and fallback order;
- pricing, quota, budgets, and remote-catalog cache/refresh policy;
- the decision to replace one immutable Registry snapshot with another.

Siumai does not define a speculative universal `ControlPlane` trait. A host resolves these concerns
before it constructs or selects a provider. A future reusable control-plane package must be justified
by multiple real adapters and must remain optional.

### Registry

Registry is a local, network-free adapter. It maps caller-owned route names to already configured
family factories and preserves the model ID remainder. Route names may encode business meaning for the
host, but Registry treats that meaning as opaque. It does not fetch model inventories, choose a
region, apply compliance policy, or select a fallback.

### Provider identity and implementation services

The public provider identity follows the canonical vendor or product boundary. An infrastructure
service name is not promoted to a parallel provider identity when it is only the implementation
behind that vendor surface.

For Alibaba:

- the public crate and provider identity are `alibaba`;
- provider options and metadata use the `alibaba` namespace;
- language, embedding, video, search, and other supported families share one configured Alibaba
  provider runtime;
- DashScope may remain in private module names, official endpoint constants, native API paths, header
  names, and wire comments;
- Siumai does not expose a separate `DashScopeProvider`, `DashScopeRoute`, deployment-scope enum, or
  region-by-model catalog.

### Model IDs and catalogs

Model IDs remain open strings/newtypes. Known model constants and private dialect rules may improve
ergonomics or encoding, but they are not allowlists. Unknown future IDs remain callable when the
selected protocol can encode the request safely.

Static provider data may describe verified protocol behavior and advisory lifecycle facts. It must
not claim to be the current account inventory or regional availability matrix. A provider-specific
remote catalog is an optional resource API; its snapshots, TTL, filtering, and routing decisions
belong to the host control plane.

### Support claims

Support claims describe technical evidence only: provider, technical platform, family, protocol/API
mode, fidelity, public stability, source, and verification date. They do not contain account, region,
or deployment availability scopes.

## Options considered

### Option A: Provider-owned static control plane

Store regions, deployment scopes, model matrices, defaults, and lifecycle state in provider crates.

Rejected because the data is mutable, account-specific, incomplete, and impossible for a released SDK
to keep authoritative. It also makes provider packages large without improving call correctness.

### Option B: A policy-rich Registry built into Siumai

Move availability, compliance, fallback, and remote discovery into the base Registry.

Rejected because it turns a useful local lookup adapter into an application platform, introduces
implicit network behavior, and forces business policy into every Siumai consumer.

### Option C: Provider execution plane plus host-owned control plane

Keep provider construction and protocol correctness in Siumai; let the host choose the configured
environment and model route.

Chosen because it keeps the base library honest, composable, offline at model-construction time, and
forward-compatible with provider changes.

## Consequences

### Positive

- Provider APIs stay small and stable while cloud inventories evolve independently.
- Unknown models and private deployments remain usable without waiting for a Siumai release.
- Registry remains deterministic, local, and independent of concrete provider crates.
- Composite providers can reuse multiple internal protocols without duplicating public identities.
- Tests focus on wire behavior and ownership boundaries instead of maintaining speculative matrices.

### Costs

- Siumai cannot promise that a model is enabled for a particular account before the provider call.
- Applications that need discovery, compliance routing, or fallback must supply that policy.
- Known-model hints are advisory and may lag the provider until updated.
- Some provider settings still contain technical location values because signing or endpoint syntax
  requires them; callers must not confuse those settings with an availability catalog.

## Migration

1. Remove `AvailabilityScope` and availability from core support claims.
2. Remove public DashScope route, deployment, workspace-routing, default-model, and regional catalog
   APIs and their facade exports.
3. Introduce `siumai-provider-alibaba` as the single composite Alibaba provider owner. Move retained
   Alibaba options, codecs, endpoint logic, and fixtures out of the generic compatible package.
4. Rename public DashScope request options and metadata to the Alibaba namespace. Do not retain
   compatibility aliases for the breaking release.
5. Keep `siumai-openai-compatible` as the shared compatible execution engine and explicit
   custom-provider escape hatch, not as the public owner of branded composite providers.
6. Keep Registry local and network-free. Host code constructs the chosen provider runtime and
   registers it under an opaque route name.
7. Move remote catalog caching, route availability, defaults, aliases, fallback, and compliance to
   host-owned code or a future optional control-plane package.
8. Add compile/dependency tests for open model IDs, Alibaba-only public identity, absence of provider
   control-plane types, and Registry independence from concrete providers.

## References

- `repo-ref/ai/packages/alibaba/src/alibaba-provider.ts`
- `repo-ref/ai/packages/alibaba/src/alibaba-chat-language-model-options.ts`
- `repo-ref/ai/packages/ai/src/registry/provider-registry.ts`
- `repo-ref/ai/packages/provider/src/provider/v4/provider-v4.ts`
- `docs/architecture/overview.md`
