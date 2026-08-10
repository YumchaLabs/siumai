# Architecture Overview

- Status: Current repository contract
- Updated: 2026-08-10
- Related decisions: `docs/adr/0010-provider-plane-and-host-control-plane.md`,
  `docs/adr/0013-provider-identity-and-family-registration.md`,
  `docs/adr/0014-canonical-language-history-and-replay.md`,
  `docs/adr/0015-validation-ownership-and-forward-compatibility.md`

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
| `siumai-openai-compatible` | one configured generic OpenAI-compatible execution engine, explicit custom-compatible escape hatches, and a bounded versioned provider-composition seam |
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

The base `Provider` trait exposes only the canonical `ProviderId`. Platform, protocol, API mode, and
provider-native replay domain belong to an exact executable `ProviderScope` carried by model
descriptors and registrations. Dated portable and native evidence uses
`SupportScope` and `NativeSupportScope` instead. A composite provider does not invent one
provider-wide scope by choosing a preferred mode or the first configured family.

Model handles are cheap values containing a model ID plus shared provider runtime. Constructing a
model does not perform remote discovery or network I/O. Unknown future model IDs remain callable;
dated model catalogs are explicit introspection, documentation, and release-freshness evidence
rather than runtime policy or closed allowlists.

Providers with multiple protocol modes expose them explicitly. One documented mode may be the
ergonomic default, while alternate modes retain distinct typed options, registrations, wire
contracts, and fidelity evidence.

A default `ProviderRegistration` may combine disjoint model families under one canonical provider
identity. Each family binding retains its own exact scope and erased model factory.
The normal path captures one configured provider, while explicit host-owned merge may combine
disjoint same-provider bindings without claiming identical credentials or runtime origin. A host can
project a combined registration to one family before assigning a route. Alternative API modes for
the same family remain separate registrations so the host chooses them explicitly.
Provider-owned profiles and support manifests describe dated evidence; Registry does not treat that
metadata as an execution allowlist and exposes no generic policy-evaluation callback.

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

## Canonical language boundary

Portable language requests use direction-aware role validation. Prefer `Message::system`,
`Message::developer`, `Message::user`, `Message::assistant`, role-specific part constructors, and
`Message::tool_result` over unchecked role/content assembly. Provider codecs validate the complete
`LanguageRequest` before transport, so response-only citations and refusals, misplaced tool results,
and unsupported request content cannot be silently accepted.

A portable `ToolCall` is always caller-executed and contains one bounded parsed JSON `ToolInput`.
Encoded function argument text is normalized exactly once by the protocol decoder. Provider-hosted
programs, custom-text calls, MCP operations, computer actions, and other provider-executed
operations remain typed native output or bounded provider-opaque replay data; they never enter the
portable runtime tool loop. A structured local function call issued by a hosted program remains a
caller-executed call, while native replay metadata preserves its caller linkage.

Generated responses do not become request history through a raw content copy. Use
`LanguageResponse::project_assistant_history()` to obtain a role-valid assistant message and
structured omissions for response-only content. Runtime records those omissions in each durable
step and uses the same projection for ordinary tool loops and structured-output repair.

## Protocol compatibility

Reuse follows this order:

1. use a provider's native protocol when it carries distinct semantics;
2. use a verified compatible engine/profile when the provider documents a compatible protocol;
3. expose an explicit custom-compatible escape hatch for caller-owned endpoints.

A branded provider owns its compatibility profile, typed options, model advice, evidence, and
fixtures. The shared engine exposes only a versioned provider-neutral codec seam and explicit
generic/custom construction. Provider-specific dialect and codec behavior remains in the branded
provider package even when execution is delegated to that shared engine.

A caller-controlled endpoint policy never proves named-provider fidelity. The generic compatible
profile may preserve a validated `EndpointConfig`, including an explicit private/shared-address
grant, but it keeps generic claims and a caller-declared custom replay audience. Responses begins
with the strict OpenAI wire baseline; documented compatible omissions require an explicit dialect
descriptor backed by fixtures.

## Canonical stream lifecycle

Direct language success has one portable termination axis: completed or incomplete. A provider
failure or cancellation returns `LanguageCallError` with a sanitized `Error` and optional bounded
non-executable partial output rather than a successful `LanguageResponse` state.

Stable language streams use one canonical event vocabulary and exactly one terminal outcome:
completed, failed, or cancelled. Failed and cancelled terminals may carry the same bounded partial
output shape as direct failures. Protocol decoders own framing-specific state, reject unexpected
EOF, preserve known-zero versus unknown usage, and never infer success from a clean transport close.
Failures reported inside an established provider stream use the failed terminal rather than a
second raw error-event lane. Protocol decoders classify only explicit, bounded wire identifiers;
transport passes bounded response diagnostics and retry hints into the decoder before body
consumption. Provider messages and raw envelopes remain available only through explicitly accessed,
bounded sensitive diagnostics.

Usage events state whether they are cumulative snapshots or deltas. Runtime reconciles observations
per provider call, treats the terminal usage as the final snapshot, and charges budgets exactly once.

When an executable item appears in both stable stream events and the terminal response, both views
must agree on item kind, identity, ownership, tool name, and normalized JSON input. A protocol may
merge a completed stable-only caller-executable function call when its terminal format permits
omission, and may accept a terminal-only item without inventing a retroactive event. Provider-native
items remain terminal-owned, and provider-only metadata does not participate in portable semantic
equality.

Opaque provider items retain provenance for same-protocol continuation. Cross-protocol projection
may emit only portable content and must reject or report loss instead of silently reinterpreting
provider-owned data or tool execution.

Replay additionally requires an exact non-secret domain on `ProviderScope`: official versus custom
audience, plus an optional caller-selected account/workspace/project/deployment label. Provider,
platform, protocol, API mode, and the whole replay domain must match; missing domains fail closed.
Registry route and model ID are not replay identities, though a protocol may impose an additional
model rule. Provider builders supply audited official audiences, while custom endpoints require an
explicit caller-declared custom audience. URLs, credentials, signed values, raw technical project or
location strings, and mutable region availability metadata are never inferred into durable replay
identity.

## Tool and option boundaries

Portable function tools live in core. Hosted tools and provider execution controls live in typed
provider options or provider-owned resource types. Tool ownership is explicit: host-executed calls
may enter the runtime approval/execution loop, while provider-executed calls remain provider-owned
events and metadata.

Common request fields contain only stable cross-provider semantics. Provider crates perform the
final deterministic merge and validation for the selected family and API mode. `CallOptions` stores
bounded ordered patches targeted to one exact configured model instance. Runtime may prepend its own
route, model, and step defaults, but those origins stay private to runtime and never become provider
API concepts. Dynamic JSON, where available, is an explicit exact-target body escape hatch;
protected authentication, endpoint, signing, transport, and canonical request fields cannot be
overridden through request options. Provider modes without a reviewed raw-body policy reject raw
options.
