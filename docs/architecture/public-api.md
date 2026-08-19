# Public API and Extension Policy

- Status: Current repository contract
- Updated: 2026-08-19

## Public entry points

Applications may depend directly on an owning crate or use the `siumai` facade:

- `Siumai::builder()` is the primary direct-application entry and produces one typed
  configured-provider hub after provider-specific required inputs are supplied;
- `Siumai<P>` exposes trait-backed family selectors, and the resulting family clients expose
  method-style portable calls plus typed `provider()` and `model()` accessors;
- direct provider crates are the authoritative surface for provider construction, protocol modes,
  typed options, metadata, and native resources;
- `siumai::providers::*` contains curated provider namespaces rather than blanket crate mirrors;
- `siumai::{language, embedding, rerank, image, speech, transcription}` owns the canonical generic
  calls, and `siumai::prelude::*` exports those modules plus their provider-neutral contracts;
- `siumai::registry`, `siumai::runtime`, and the curated `siumai::transport` configuration
  namespace are optional facade integrations;
- transport execution, protocol codecs, MCP, and server adapters remain available from their
  owning packages instead of being relayed through broad facade namespaces.

The facade does not expose a universal client, provider enum, capability matrix, or provider
downcasts. Typed family clients and routed models implement the same family traits, so an
application can choose direct ergonomics, provider fidelity, local routing, or a combination
without maintaining two execution APIs.

## Typed direct facade and canonical family seam

`Siumai<P>` is a typed hub around one concrete configured provider. It owns no default-model slots,
hidden Registry, route policy, runtime, or business fallback. A family selector is callable only
when the provider implements the corresponding family-provider trait; unsupported families fail at
compile time instead of returning a runtime capability error.

The bound clients use one stable vocabulary:

| Family client | Simple operations | Advanced operation |
|---|---|---|
| `LanguageClient` | `generate`, `stream` | `call` |
| `EmbeddingClient` | `embed` | `call` |
| `RerankClient` | `rerank` | `call` |
| `ImageClient` | `generate` | `call` |
| `SpeechClient` | `synthesize` | `call` |
| `TranscriptionClient` | `transcribe` | `call` |

These methods are ergonomic delegates, not a second execution implementation. They preserve the
inner model descriptor, canonical route, exact configured instance, family limits, provider-option
preflight, complete response, and stream lifecycle. `provider()` returns the retained concrete
provider and `model()` returns the exact provider-owned model; no arbitrary public provider/model
pair constructor exists.

The facade exposes six root family modules. Each module keeps the existing family request and
response contract rather than wrapping results in a universal response:

| Family module | Default operation | Bound call terminal |
|---|---|---|
| `language` | `generate`, `stream` | `call(...).generate()`, `call(...).stream()` |
| `embedding` | `embed` | `call(...).embed()` |
| `rerank` | `rerank` | `call(...).rerank()` |
| `image` | `generate` | `call(...).generate()` |
| `speech` | `synthesize` | `call(...).synthesize()` |
| `transcription` | `transcribe` | `call(...).transcribe()` |

Language entry points accept `LanguageInput`, which normalizes a string, one `Message`, a message
list, or a complete `LanguageRequest` immediately and losslessly into the canonical request. The
other families retain their complete request types because their required inputs and validation are
not interchangeable.

Every `call` builder borrows one already selected live model and owns one request, one replaceable
`CallOptions` baseline, and ordered typed provider-option patches. `with_options` and
`with_provider_options` are fallible and validate their complete candidate state synchronously.
Execution resolves the deadline, validates the portable request, validates exact provider-option
selection for that same model, and dispatches once. The builder never stores a Registry or route
string and never re-resolves the target after options are bound.

A typed family client, concrete provider model, and Registry-resolved `Arc<dyn LanguageModel>` (or
the corresponding family trait object) use the same root entry point. Registry resolution is
explicit and network-free; `Siumai::builder()` does not register providers, select routes, match
provider identities, or downcast erased models. Applications retain the configured concrete
provider beside Registry when they also need native resources or sessions.

Language calls return the complete `LanguageResponse` or established `LanguageStream`.
`LanguageResponse::text_parts()` and `output_text()` are display-oriented projections only:
reasoning, refusals, tools, citations, media, and provider-native state remain on the complete
response. Text projection is not assistant-history projection; use
`project_assistant_history()` for continuation and replay.

## Provider construction

Facade construction starts from a zero-state builder and one zero-argument product selector such as
`.openai()`, `.anthropic()`, or `.gemini()`. Provider-specific typed stages require credentials,
profiles, projects, locations, or other constructor inputs in the real builder's order. `.build()`
is unavailable until the required sequence is complete, remains synchronous and network-free, and
returns the provider-owned configuration error. Optional configuration passes through one
`configure_provider` closure over the real builder; the facade does not mirror its setters or retain
a second plaintext credential copy.

Provider builders configure credentials, endpoint policy, one
`ProviderHttpTransportSettings` snapshot, and provider-wide typed defaults. They do not select an
application route, discover models, or perform network I/O during construction. Every configured
provider and compatibility engine accepts stateless HTTP infrastructure through
`with_http_transport_settings(...)`; the former duplicated limits/retry/connect/call/read/observer
setters are not compatibility aliases.

Provider methods construct lightweight family models synchronously from open model IDs. When a
provider supports multiple language protocols, the provider exposes named constructors such as
`chat_completions(model)` or `responses(model)` and a documented `language(model)` default.

The facade fixes OpenAI `.language(model)` to Responses and exposes
`.chat_completions(model)` as the explicit alternative. Gemini `.language(model)` is fixed to
Interactions and `.generate_content(model)` is the explicit alternative. These choices are part of
the public contract and never depend on model-name patterns.

The base `Provider` trait exposes only `provider_id()`. Inspect a concrete model descriptor or an
explicit family registration when platform, protocol, or API-mode identity matters. Those values
describe one execution surface and cannot truthfully summarize a composite provider.

Dynamic registration starts with one `ProviderRegistration::from_*` family binding and may add
disjoint families with `bind_*`. Each binding owns only its exact scope and erased factory. An empty
registration is not constructible, and `for_family` can narrow a combined registration before the
host assigns a route.
Alternative modes for one family use the provider's mode-specific registrations and distinct
Registry routes. Facade `register_provider` returns a typed error when a valid provider configuration
has only provider-native resources or jobs and therefore no portable family registration.

Registry does not evaluate lifecycle, allowlist, or model-capability policy. A host that needs those
decisions keeps the concrete provider or its support manifest beside the registration, evaluates the
evidence before resolution, and owns any warning or rejection. Unknown and retired model IDs remain
constructible; the concrete request planner rejects only stable technical constraints that it can
prove locally.

Registry adds canonical route projection through private family wrappers and exposes
`RegistryModelContext` for typed resolution errors. It does not expose an execution-decoration
pipeline. A host-owned family-trait decorator must preserve the complete `ModelDescriptor` and
delegate `route_id()` so route-bound provider options and runtime defaults continue to match the
configured target.

## Provider HTTP transport and call controls

`ProviderHttpTransportSettings` is owned by `siumai-transport` and is reachable through
`siumai::transport` when the facade `transport` feature is enabled. The curated namespace contains
endpoint policy, limits, retry/settings, observer/event, and configuration-error values. It does
not expose `ProviderTransport`, authentication appliers, request plans, raw responses, resource
downloaders, or socket types.

The settings value applies only to provider-owned stateless HTTP APIs. Endpoint/authentication,
provider retry classification, and per-request replay proof remain separate authorities. Provider
options and request bodies cannot override those authorities. Provider WebSocket/Realtime sessions,
media jobs, and external downloads retain their own lifecycle-specific Direct controls. MCP owns a
separate configuration that may reuse only the route value.

`CallOptions::with_timeout` starts at the outer logical invocation and resolves once to an absolute
deadline; an existing earlier deadline wins. `with_max_attempts` caps total attempts for one
provider HTTP call but cannot promote `ReplaySafety::Never` or expand provider retry policy.
`without_retry()` remains the one-attempt convenience.

Transport observers receive only structural attempt-loop data and an opaque correlation token.
Observation ends at bounded response return or byte-stream establishment, not response-body or
protocol completion. Provider/route/account attribution is supplied by a host-authored observer
wrapper and never enters transport events.

`HttpTransportRoute::Direct` is the default and remains independent of environment/system proxy
configuration. A custom endpoint is a reverse-gateway destination, not a forward proxy. Callers
select one explicit forward proxy with
`HttpTransportRoute::trusted_connect(ProxyEndpoint)` and apply it through
`ProviderHttpTransportSettings::with_route(...)`. The destination must be a public HTTPS origin.
Public proxy origins must also use HTTPS; explicitly granted local HTTP proxies are allowed only
without authentication.

`ProxyBasicCredential` is bounded, header-safe, redacted, and bound only to the proxy audience.
Provider authentication remains bound to the provider audience and is applied only inside the
CONNECT tunnel after destination TLS verification. Credentials are immutable configuration
snapshots: rotate proxy Basic credentials by rebuilding the configured provider rather than by
mutating a live client or registering a refresh callback.

Direct mode resolves and validates the provider endpoint and peer. Trusted CONNECT mode resolves
and validates the proxy endpoint and peer, transferring destination DNS/peer selection to that
trusted proxy while preserving the logical provider URL, inner TLS hostname/certificate,
credential audience, redirects, replay proof, deadlines, and bounds.

Streamable HTTP MCP exposes the same route types from `siumai-mcp` and accepts the route through
`McpClientConfig::with_http_transport_route(...)`. It does not import provider settings,
credentials, authentication, retry policy, or execution transport. Stdio MCP is unaffected.

This surface does not include environment discovery, SOCKS or PAC, named proxy-product
certification, opaque or bearer/Negotiate/NTLM/Kerberos proxy authentication, URL-userinfo
credentials, raw client/custom-fetch injection, custom proxy CA or mTLS configuration, WebSocket,
Realtime, provider-returned external downloads, or arbitrary-URL proxy routing.

## Typed provider extensions

Provider-specific request behavior uses types owned by the provider package. A typed call option
declares its provider namespace, model family, and API mode and validates before type erasure. The
canonical facade path attaches it with
`family::call(&model, request).with_provider_options(&options)?`; lower-level callers may assemble
the same exact-target patch through `CallOptions::with_provider_options_for(&model, &options)`.
Both paths bind normal provider intent to one configured provider instance. A provider option type
may opt into reusable unbound targeting only after its author proves that it carries no credentials,
replay state, or instance-sensitive body data.

Runtime may prepend route, model, and step defaults internally, but those host-level origins are not
part of the provider-facing contract. Providers receive one ordered exact-target selection, apply
typed patches in order, then either apply one explicitly supported raw body overlay or reject raw
options. Raw options are also bound to an exact model instance and can never alter authentication,
endpoints, signing, transport policy, or protected canonical request fields.

Provider behavior attached to one message, content part, or tool definition uses a typed durable
annotation stored beside that semantic node. Annotations have no precedence or recursive merge
algorithm. Foreign history annotations remain inert; a provider reads only its exact namespace,
API mode, and node target. Provider-native state required for faithful replay remains a
bounded provenance-bearing opaque item rather than an optional annotation.

An OpenAI prompt-cache content annotation expresses one provider wire breakpoint only. It does not
predict a cache read or write, classify history, or select a provider-side lookback window. Request
mode, TTL, retention, and cache key remain separate provider-owned options; see ADR 0016.

Use a common request field only when its semantics are stable across providers. Examples of
provider-owned behavior include prompt-cache controls, reasoning modes, hosted search, MCP or code
execution tools, service tiers, provider-specific log probabilities, and resource references.

Provider metadata follows the same ownership rule. Shared response and usage fields remain neutral;
typed provider metadata views expose provider-specific details without moving commercial provider
types into core.

Provider-owned profiles or support manifests expose dated claims for named surfaces. They are
introspection and maintenance evidence, not capability gates: custom endpoints may carry generic or
empty evidence while still constructing models that the selected protocol can encode safely.

`OpenAiProvider` custom endpoints still assert the OpenAI wire baseline. A relay that intentionally
deviates from that baseline uses `OpenAiCompatibleProfile::custom_endpoint`, an explicit custom
replay audience, and a caller-selected `ResponsesWireDialect` only when fixtures prove the allowed
omissions. Transport policy labels never promote a generic relay into a named-provider claim.

## Provider-owned sessions

Persistent or bidirectional provider workflows remain outside the portable model-family traits when
their lifecycle is not genuinely shared. Their public API follows the provider product instead of a
universal session command envelope.

OpenAI Responses WebSocket is an experimental provider-owned session enabled separately with
`openai-responses-websocket`. Callers acquire it from a Responses model, connect once, and then run
generated turns or native-only warm-up turns:

```rust,ignore
let model = provider.responses("gpt-5.6")?;
let config = model.websocket()?;
let session = config.connect(CallOptions::default()).await?;

let turn = session
    .generate(LanguageRequest::new(vec![Message::user("Continue the task")]), CallOptions::default())
    .await?;
```

The session permits one active response at a time, reuses the Responses semantic decoder for every
turn, and exposes exact native events alongside portable projections. `generate: false` warm-up is
native-only and never fabricates a portable language response. The official OpenAI provider may use
the provider-owned default WebSocket endpoint; a custom HTTP provider must configure its WebSocket
endpoint explicitly and never inherits the official support claim.

Each turn has one provider-private lifecycle owner for queue admission, submission certainty,
response identity, terminal publication, fallback failure, consumer EOF, and actor-exit cleanup.
Public turn handles expose read-only state projections; they do not own an independent settlement
flag. Production WebSocket transport and deterministic test adapters are private implementation
details rather than application extension points.

This shape is intentionally not a new portable `SessionModel` family. Other provider sessions may
share transport or lifecycle helpers internally while retaining their own typed commands, events,
and settlement rules.

## Stability

The public contract has three practical levels:

1. stable typed-facade and family contracts, including the documented `Siumai::builder()` chains,
   family binding and operation names, provider/model accessors, explicit alternate-mode selectors,
   six root family modules, bound call entry points, complete response types, and typed
   provider-option ownership;
2. provider-owned stable APIs for documented provider capabilities and resources;
3. explicitly named experimental modules for sessions, jobs, or capabilities whose lifecycle is
   not yet a stable family primitive.

Compatibility namespaces, old generic builders, protocol relays, and source-layout aliases are not
a stability tier. Breaking releases delete them after the replacement path and migration guidance
exist.

The typed hub and retained root family seam form the compatibility boundary established by
[ADR 0020](../adr/0020-typed-siumai-provider-hub.md), which supersedes ADR 0019's rejection of a
`Siumai` entry while preserving its family-call ownership. A later beta change to a provider
selector, family binding, canonical/alternate mode, operation, root module, bound call, result type,
or typed extension owner requires an explicit
architecture rationale, an exact old-to-new symbol map, synchronized README/rustdoc and
compile-checked examples, a changelog entry, and release review naming the affected symbols. The
project may still make an evidence-backed beta break; it must not rename the primary journey
silently or preserve contradictory aliases as a substitute for migration evidence.

## Features

Facade provider features activate the selected optional provider dependency plus the narrow
`transport` configuration feature. They do not activate unrelated providers, runtime, Registry,
MCP, Realtime, or Responses WebSocket code. Provider package features represent real compile-time
behavior, such as an optional protocol or message capability; empty relay features are removed.

Applications that need the full implemented provider-owned surface should depend on the provider
crate directly. A facade feature must not activate unrelated providers, protocols, or job/session
integrations.

## Errors, streams, and cancellation

Family calls return sanitized errors with operation/provider/model context, retry hints, and bounded
provider details where available. Language direct calls use `LanguageCallError`, which contains the
canonical `Error` plus an optional bounded, non-executable `PartialLanguageOutput`. Other family
calls return the canonical `Error` directly. Credentials, signed URLs, raw headers, and unbounded
response bodies never appear in ordinary `Debug` or display output.
`ErrorKind::ContextWindowExceeded` and `ErrorKind::Unavailable` distinguish exact provider signals
that callers commonly handle differently from invalid input or an unknown provider failure. A
provider message is never inspected heuristically to infer either category.

`LanguageResponse` has one success termination axis: `LanguageTermination::Completed` or
`LanguageTermination::Incomplete`. Provider-returned failure and cancellation are not successful
response states. Established streams settle exactly once through `StreamTerminal::{Completed,
Failed, Cancelled}`; failed and cancelled terminals may carry the same bounded partial-output shape
as a direct `LanguageCallError`. Usage observations are explicitly `Snapshot` or `Delta`, so runtime
can reconcile late usage-only frames without double charging a call.

Language streaming begins only after the provider stream is established. The stream owns its
transport resources and cancellation child; dropping it releases those resources. Consumers must
observe a terminal completed, failed, or cancelled event. Protocol/server encoders reject events
after terminal and do not manufacture success on unexpected EOF. In-band provider errors are failed
terminals carrying the same typed `Error` contract as setup failures; raw provider error JSON is not
a second high-level failure channel.

Provider-owned persistent turns follow the same settlement rule: each established turn produces one
canonical terminal outcome or a typed error. Unexpected EOF, malformed frames, queue exhaustion,
deadline expiry, and cancellation are explicit outcomes. A turn-local provider failure may leave a
synchronized session reusable; transport or protocol desynchronization closes it conservatively.
Standard retryable WebSocket availability closes, including service restart, try-again-later, and
bad-gateway codes, retain `ErrorKind::Unavailable` without exposing the provider-controlled close
reason.

## Documentation rule

Examples and README snippets use only current public paths and declare their required features.
Named provider support claims include an official source, verification date, family/API-mode or
native-surface scope, fidelity, and stability through a provider-owned profile or support manifest.
Model constants are completion hints and dated evidence, not a closed union of every remotely
available model and never a runtime callability gate.
