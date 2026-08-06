# Architecture Overview

- Status: Current repository contract
- Updated: 2026-08-05
- Related decisions: `docs/adr/0010-provider-plane-and-host-control-plane.md`,
  `docs/adr/0013-provider-identity-and-family-registration.md`

## Product shape

Siumai is a Rust-first workspace for connecting applications to AI model providers. It offers two
complementary paths over the same underlying model contracts:

- provider-owned APIs expose faithful protocol modes, typed options, metadata, and resources;
- provider-neutral family traits, Registry, runtime helpers, and the `siumai` facade provide
  portability where the behavior is genuinely shared.

The unified interface is an ergonomic assembly layer. It is not a universal client, a capability
bag, a remote model catalog, or a replacement for provider-specific APIs.

## Stable and experimental operations

The stable provider-neutral surface contains six one-call model families:

| Family | Primitive |
|---|---|
| Language | one generated response or one established event stream |
| Embedding | one request containing one or more inputs |
| Rerank | one query and one candidate set |
| Image | one final-result generation call |
| Speech | one final-result synthesis call |
| Transcription | one final-result transcription call |

Realtime sessions, live connections, streaming transcription or translation, video generation,
and asynchronous media jobs have distinct lifecycle semantics and remain explicit experimental or
provider-owned resources. Files, batches, skills, assistants, hosted applications, and remote model
catalogs are resources rather than model families.

## Workspace layers

| Layer | Ownership |
|---|---|
| `siumai-core` | provider-neutral identities, family traits, requests, responses, usage, errors, options, and canonical stream lifecycle |
| `siumai-transport` | HTTP/WebSocket execution, endpoint policy, authentication application, redirects, replay safety, retries, deadlines, cancellation, and resource bounds |
| `siumai-protocol-*` | wire schemas, request/response codecs, SSE or WebSocket state machines, and protocol-owned metadata projection |
| `siumai-openai-compatible` | one configured OpenAI-compatible execution engine, verified profiles, and explicit custom-compatible escape hatches |
| `siumai-provider-*` | provider construction, credentials, technical endpoints, API modes, typed options, model advisories, provider codecs, and native resources |
| `siumai-registry` | immutable, network-free lookup from host-owned route IDs to configured provider registrations |
| `siumai-runtime` | provider-neutral tool loops, structured output, approvals, budgets, and durable multi-step execution |
| `siumai-mcp` | MCP client/server integration and MCP-specific lifecycle/security policy |
| `siumai-server` | server and gateway adapters over runtime, core, and protocol contracts |
| `siumai` | curated facade, feature aggregation, prelude, Registry adapters, and primary ergonomic entry points |

Dependency direction flows from facade and integrations toward provider/runtime/registry, then into
core, protocol, and transport owners. Provider crates do not depend on the facade or Registry, and
Registry remains free of provider implementations and network execution.

Wire projection stays with the protocol codec or the concrete server/gateway integration that
exposes that wire contract. Siumai does not maintain a general cross-protocol bridge package: a
generic bridge would duplicate codec state machines and invent a global loss model before real
consumers have demonstrated one. A future shared projection module must be extracted from multiple
working integrations, remain canonical-only, and make loss and resource bounds explicit.

## Configured providers and model handles

A configured provider owns long-lived shared runtime state: credentials, endpoint policy,
transport, retry policy, concurrency limits, protocol profiles, and provider resources. Provider
construction is synchronous and model-independent.

The base `Provider` trait exposes only the canonical `ProviderId`. Platform, protocol, and API mode
belong to an exact executable `ProviderScope` carried by model descriptors, policy contexts, and
registrations. Dated portable and native evidence uses `SupportScope` and `NativeSupportScope`
instead. A composite provider does not invent one provider-wide scope by choosing a preferred mode
or the first configured family.

Model handles are cheap values containing a model ID plus shared provider runtime. Constructing a
model does not perform remote discovery or network I/O. Unknown future model IDs remain callable;
dated model catalogs are advisories used for exact known-policy checks, documentation, and release
freshness rather than closed allowlists.

Providers with multiple protocol modes expose them explicitly. One documented mode may be the
ergonomic default, while alternate modes retain distinct typed options, registrations, request
policy, and fidelity evidence.

A default `ProviderRegistration` may combine disjoint model families under one canonical provider
identity. Each family binding retains its own exact scope, request policy, and erased model factory.
The normal path captures one configured provider, while explicit host-owned merge may combine
disjoint same-provider bindings without claiming identical credentials or runtime origin. A host can
project a combined registration to one family before assigning a route. Alternative API modes for
the same family remain separate registrations so the host chooses them explicitly.
Provider-owned profiles and support manifests describe dated evidence; Registry does not treat that
metadata as an execution allowlist.

## Provider plane and host control plane

Provider crates own technical facts required to execute an API request:

- canonical vendor/product identity;
- authentication and endpoint construction;
- protocol selection and wire behavior;
- typed provider options and metadata;
- provider-native resources;
- exact request-time validation and dated support evidence.

The host application owns business and deployment policy:

- tenant, account, project, workspace, and route selection;
- region and data-residency policy;
- deployment inventory and model allowlists;
- defaults, aliases, pricing, quota, health, weights, and fallback.

Technical coordinates such as an AWS signing region, Vertex project/location, Azure deployment, or
custom base URL may be required inputs to a provider builder. Siumai validates and uses them but
does not choose them, infer account availability, or publish mutable regional inventories as stable
runtime types.

## Protocol compatibility

Reuse follows this order:

1. use a provider's native protocol when it carries distinct semantics;
2. use a verified compatible engine/profile when the provider documents a compatible protocol;
3. expose an explicit custom-compatible escape hatch for caller-owned endpoints.

A compatibility profile states only the fidelity that fixtures and official documentation prove.
Provider-specific policy remains in the branded provider package even when execution is delegated
to a shared protocol engine.

## Canonical stream lifecycle

Stable language streams use one canonical event vocabulary and exactly one terminal outcome:
completed, failed, or cancelled. Protocol decoders own framing-specific state, reject unexpected
EOF, preserve known-zero versus unknown usage, and never infer success from a clean transport close.

Opaque provider items retain provenance for same-protocol continuation. Cross-protocol projection
may emit only portable content and must reject or report loss instead of silently reinterpreting
provider-owned data or tool execution.

## Tool and option boundaries

Portable function tools live in core. Hosted tools and provider execution controls live in typed
provider options or provider-owned resource types. Tool ownership is explicit: host-executed calls
may enter the runtime approval/execution loop, while provider-executed calls remain provider-owned
events and metadata.

Common request fields contain only stable cross-provider semantics. Provider crates perform the
final deterministic merge and validation for the selected family and API mode. Dynamic JSON, where
available, is an explicit checked escape hatch; protected authentication, endpoint, and transport
fields cannot be overridden through request options.
