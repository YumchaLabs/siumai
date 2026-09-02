---
title: Transport Ergonomics and Middleware Boundaries - Plan
type: refactor
date: 2026-08-16
deepened: 2026-08-16
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Transport Ergonomics and Middleware Boundaries - Plan

## Goal Capsule

| Field | Contract |
|---|---|
| Objective | Make retry, timeout, proxy, limits, and transport observation straightforward to configure across Siumai without weakening replay safety, endpoint isolation, credential authority, resource bounds, or provider ownership; delete the speculative Registry-only middleware surface instead of adding another extension system. |
| Priority | P0: preserve transport and replay authority. P1: unify Direct-mode provider HTTP settings, call-level timing/retry ergonomics, observer completeness, facade reachability, and Registry middleware cleanup. P2: add the explicit trusted HTTP proxy route and MCP route adapter as a separately mergeable milestone. |
| Compatibility posture | Breaking beta changes are allowed. Delete duplicated builder setters, unused middleware APIs, stale exports, and tests that encode the wrong ownership instead of retaining compatibility aliases. |
| Evidence hierarchy | Current repository architecture and executable contracts define internal ownership. Official framework and provider documentation defines external behavior. The local Vercel AI SDK checkout is secondary prior art for ergonomics and edge cases, not an architecture target. |
| Architectural boundary | Transport owns network route, retries, replay admission, timeouts, limits, endpoint enforcement, and attempt observation. Providers own credentials, endpoints, replay declarations, retry classification, and wire semantics. Registry owns deterministic lookup only. Runtime owns multi-step policy. The host owns business routing, fallback, caching, pricing, quota, compliance, and telemetry installation. |
| Execution posture | Implement in dependency order with serial Cargo commands and the shared target directory. Scoped Conventional Commits are allowed after green unit gates. Do not publish, push, or open a PR unless separately authorized. |
| Stop conditions | Stop for an unavoidable weakening of credential audience or replay proof, a proxy design that cannot keep provider and proxy credentials separate, a dependency cycle, or overlapping user changes that cannot be isolated safely. |

---

## Product Contract

### Summary

Siumai already has the stronger infrastructure model: explicit replay proof, guarded endpoints and DNS, credential audience isolation, bounded bodies and streams, cancellation-aware retries, and sanitized diagnostics. Its weakness is configuration ergonomics. Provider builders repeat the same transport fields under inconsistent names, most transport configuration types are not reachable through the facade, call-level timeout and retry controls are too coarse, only OpenAI exposes the transport observer, and enterprise forward proxying has no safe explicit path.

The repository also already exposes `RegistryMiddleware`, but it is Registry-only, undocumented in the Registry architecture contract, has no production consumer, spans all six model families without evidence, and validates identity with equality that excludes configured-instance capability. Rather than copy the AI SDK's experimental, broad middleware hooks, this plan removes that speculative surface. Ordinary Rust model wrappers remain possible, and a future first-class model layer requires proven consumers and a separately reviewed contract.

The target is a smaller and more coherent public surface:

- one transport-owned `ProviderHttpTransportSettings` value used by every configured provider stateless HTTP runtime;
- one explicit direct-or-trusted-proxy network route, with no environment discovery or raw client injection;
- one relative timeout and one caller attempt cap that only narrow existing authority;
- one payload-free attempt observer available consistently across providers;
- one curated facade path for the configuration types that provider builders already expose;
- no Registry middleware framework, transport interceptor, global telemetry registry, or host-policy engine.

### Problem Frame

The current code has three forms of accidental complexity.

First, transport configuration is duplicated across provider and compatibility builders. The duplication is not harmless: method names differ, some builders omit retry or observer controls, composite providers manually copy fields into several internal engines, and OpenAI provider-level settings mix stateless HTTP and provider-owned WebSocket behavior.

Second, callers cannot express several common intents cleanly. `CallOptions` accepts an absolute `Instant` but no relative timeout. Per-call retry control is all-or-nothing even though a caller often wants to cap attempts below a provider default. Forward proxy use requires abandoning the guarded transport entirely, which is intentionally impossible today.

Third, middleware is in the wrong place. `RegistryMiddleware` turns deterministic lookup into an execution decoration seam, applies only to Registry-resolved models, has no direct-provider parity, and grants wrappers enough authority to implement hidden retry, fallback, caching, or request rewriting. AI SDK demonstrates that this broad shape is convenient, but its experimental middleware also permits provider/model identity overrides, generate/stream substitution, and heuristic transformations. Those are specifically outside Siumai's intended boundary.

### AI SDK and Tower Comparison

| Concern | Reference approach | Siumai decision | Reason |
|---|---|---|---|
| Per-call retry ergonomics | AI SDK exposes a caller-facing retry count. | Expose a caller attempt cap, then intersect it with provider policy and `ReplaySafety`. | A convenience count may narrow retries but cannot prove that replay is safe. |
| Request timing | AI SDK relies on abort/cancellation signals and provider/runtime options. | Add one relative timeout that resolves to one absolute deadline at the outer Siumai-owned call boundary. | One deadline composes across queueing, backoff, stream establishment, runtime steps, and structured repair without restarting. |
| Network customization | AI SDK providers commonly permit custom fetch/client injection. | Expose a closed Direct-or-trusted-CONNECT route only. | Raw client/fetch injection can bypass endpoint, credential, replay, redirect, retry, and bounds authority. |
| Model middleware | AI SDK's experimental middleware can transform parameters/results and substitute generate/stream behavior. | Delete Registry middleware and keep ordinary explicit Rust trait decorators as host code. | The repository has no proven built-in consumer, and the broad hook can absorb fallback, caching, heuristics, and identity changes that Siumai does not own. |
| Middleware composition | Tower provides a mature `Service`/`Layer` stack with explicit order and readiness. | Do not expose Tower or a parallel layer stack in this plan. | Siumai's six async family traits and provider authority boundaries do not yet have proven consumers for one shared service type; transport observation and explicit wrappers cover the current needs. |
| Telemetry | AI SDK's higher-level telemetry can carry logical operation metadata and optionally content. | Keep a synchronous payload-free HTTP attempt observer. | Transport can truthfully explain attempts and establishment, but model/run semantics and content capture belong to a later host adapter. |
| Registry | AI SDK's provider registry is an ergonomic provider/model lookup surface. | Keep Siumai Registry deterministic and execution-policy free. | Lookup, route context, and exact provider-option targeting are useful; hidden execution decoration is not required for them. |

### Requirements

#### Unified provider HTTP transport settings

- R1. `siumai-transport` owns one cloneable `ProviderHttpTransportSettings` value. It is the single public carrier for provider stateless-HTTP limits, provider retry policy, connect/call/read timeouts, payload-free attempt observer, and explicit HTTP network route.
- R2. The settings value validates zero, overflow, and hard resource limits before client construction; its `Debug` output exposes only structural values and whether optional observer/proxy credentials are configured.
- R3. Endpoint, credential/auth applier, DNS resolver, provider retry classifier, replay safety, request target, headers, and body remain outside the settings value because their owner or lifecycle differs.
- R4. Every configured provider and both compatibility engines accept the same HTTP settings value through `with_http_transport_settings(...)`. Within one configured provider, branches with the same exact endpoint, credential audience/auth owner, HTTP route, and settings clone one shared `ProviderTransport`; branches with a distinct endpoint, audience, signing owner, or network mechanism use a separate transport. Composite providers make this topology explicit without inferring capabilities from provider/model names.
- R5. The duplicated provider-level setters for HTTP limits, retry, HTTP timeouts, and transport observer are deleted after migration. Provider-owned Realtime, Responses WebSocket, media-session, and external-download timeout/limit APIs remain only where their lifecycle differs materially. The migration must replace any current accidental inheritance from generic HTTP setters with explicit lifecycle-owned controls before deleting those setters; it must not silently reset those surfaces to defaults.
- R6. Custom endpoint or base URL configuration remains a reverse-gateway destination decision and continues to require exact endpoint policy and replay audience. It is never renamed or treated as a forward proxy.
- R7. No universal provider builder trait, source-text validation script, provider capability matrix, or second hard-coded provider inventory is introduced to enforce consistency.

#### Call-level timeout and retry intent

- R8. `CallOptions` supports a validated relative timeout in addition to an absolute deadline. A relative timeout starts when the outer logical family call or runtime run accepts the options, not when the options value was constructed.
- R9. Relative timeout resolution produces one absolute deadline and clears the relative form. Every Siumai-owned public model-family implementation and wrapper performs the same idempotent resolve-once operation, including facade helpers, Registry wrappers, compatibility engines, configured provider models, and runtime steps; retries, backoff, authentication refresh, stream establishment, and nested provider calls propagate that same deadline and never extend or reset it. Third-party trait implementations receive a public documented resolution helper and are responsible for invoking it at their own outer call boundary; Siumai does not claim that an open trait can enforce this behavior in external implementations.
- R10. When both a relative timeout and absolute deadline are present, the earliest resulting deadline wins. Zero or platform-unrepresentable durations fail with a typed, matchable configuration/input error before network submission.
- R11. Caller retry intent caps total attempts for each logical provider HTTP call below the provider policy. The effective attempt budget is the minimum of provider policy and caller cap; a runtime model step receives its own call budget rather than sharing one counter across the run, streaming uses the cap only before establishment, and a one-attempt cap replaces the current `without_retry` special case.
- R12. A caller cap never promotes `ReplaySafety::Never`, never creates an idempotency key, and never adds a retryable status. Operation replay proof remains the first authority.
- R13. `RetryPolicy` owns an explicit `max_server_delay` for standard `Retry-After` advice, separate from local exponential `max_backoff`. Advice beyond that ceiling or the remaining deadline stops retrying and returns the current typed failure; Siumai does not retry earlier than the server requested.
- R14. Retry explainability is structural and payload-free: observation exposes configured versus effective attempt budgets, the exact limiting authority, retry reason, delay, and final attempt-loop outcome. No error-message parsing or provider-name heuristic is used.

#### Explicit trusted HTTP proxy route

- R15. Direct networking remains the default. Direct mode continues to disable environment/system proxies, automatic redirects, automatic referer forwarding, and reqwest retry.
- R16. The only new forward-proxy capability is an explicit trusted HTTP CONNECT route selected by the caller. It is a transport-owned typed value, not a URL callback, raw `reqwest::Client`, custom fetch, or middleware hook.
- R17. The proxy endpoint has its own validated origin/policy and an optional typed `ProxyBasicCredential` containing a bounded, header-safe username and secret password. V1 does not accept an opaque `Proxy-Authorization` value, bearer token, Negotiate/NTLM callback, or URL userinfo. Proxy credentials are accepted only for an `https://` proxy; an explicitly granted cleartext `LocalExplicit` proxy must be unauthenticated. Proxy credentials are bound to the proxy audience, provider credentials remain bound to the provider audience, and neither credential set may overwrite or leak into the other request phase. The credential is an immutable configuration snapshot; rotation is performed by rebuilding the configured provider or MCP client rather than by adding a refresh callback or global credential registry.
- R18. The first proxy slice tunnels only HTTPS destination origins. It rejects provider `LocalExplicit` targets and other non-tunneled destination shapes before network submission.
- R19. Trusted proxy mode explicitly transfers destination DNS resolution and peer-address enforcement to the selected proxy. Proxy mode resolves and validates the proxy endpoint/peer through a route-specific resolver rather than reusing the Direct origin resolver; Siumai still validates the logical destination shape, provider credential audience, TLS hostname/certificate, request target, redirect policy, replay proof, deadlines, and resource bounds.
- R20. The shared proxy route type is consumed by authenticated provider HTTP transport and streamable HTTP MCP. MCP retains its own endpoint policy, authentication, limits, lifecycle, and never-replay authority; it does not consume `ProviderHttpTransportSettings` or `ProviderTransport`. This preserves agent context/action parity without creating a new MCP retry or provider abstraction.
- R21. The settings type is explicitly scoped to provider-owned stateless HTTP API transport. Provider-returned external resource downloads and provider WebSocket/Realtime sessions keep their independent configuration and remain direct-only in this plan. No implementation silently treats the HTTP route as a global provider route or propagates proxy state solely to manufacture an unsupported-session error.
- R22. SOCKS, PAC files, environment proxy discovery, proxy-selection closures, raw HTTP interception, custom CA stores, mTLS, transparent local gatewaying, and proxy support for arbitrary resource URLs are explicitly deferred. The v1 claim is bounded HTTPS CONNECT behavior proven by deterministic protocol/security fixtures, not certification for every enterprise proxy appliance; named deployment support requires separate dated evidence and may not become a release-gating live test.

#### Attempt observation and telemetry boundary

- R23. Every configured provider can install the same transport observer through the unified settings value. No provider-specific observer setter or OpenAI-only path remains. Transport itself does not invent provider, model, route, account, or tenant identity; a host that needs attribution installs a small observer wrapper per configured provider and attaches its own context outside the payload-free transport event.
- R24. Transport events describe the HTTP attempt loop precisely: a payload-free per-call correlation token, attempt budget resolution, attempt start, a response-head-received lifecycle marker carrying only R14's structural fields and no raw headers, retry scheduling/decline, response return or stream establishment, and pre-return/establishment failure, cancellation, or timeout. Observation ends when a buffered response is returned or a byte stream is established; it does not claim to observe response-body consumption or protocol completion. A repository-owned conformance fixture proves that host-supplied provider context can be correlated without adding provider identity to the transport schema.
- R25. Observation remains read-only, synchronous, bounded, and payload-free. It exposes no URL, query, header values, credential material, request/response body, prompt, output, tool input/output, or provider error body.
- R26. This plan does not add OpenTelemetry dependencies, a global observer registry, subscriber/exporter initialization, default prompt/output capture, or a model/runtime telemetry framework. A future adapter may consume the stable observer surface after a real consumer proves the mapping.

#### Middleware and Registry boundary

- R27. Delete public `RegistryMiddleware`, `RegistryBuilder::middleware`, `RegistrySnapshot::middlewares`, the middleware stack, facade/prelude exports, and tests that treat Registry as an execution decoration pipeline.
- R28. Registry continues to wrap resolved models only to attach canonical route context and error context. Public `RegistryModelContext` and typed resolve-error context remain available; only the route model wrappers become private. Registry remains immutable, network-free, and free of execution policy.
- R29. Do not replace the deleted API with a core middleware trait, hook map, transform callback, transport interceptor, or built-in wrapper collection in this plan. Rust users can still implement an ordinary family-trait decorator explicitly.
- R30. A future first-class model layer requires at least two repository-owned consumers that cannot be served by transport observation, provider options, request construction, runtime orchestration, or an ordinary wrapper. It must be family-specific, explicit, outside transport retry, and unable to alter provider instance, route, endpoint, authentication, replay, or provider wire authority.

#### Facade, documentation, and cleanup

- R31. The facade gains an additive `transport` feature and curated `siumai::transport` configuration namespace. Provider facade features activate it. Milestone A re-exports endpoint policy, limits, retry/settings, observer/event, and configuration-error types; Milestone B adds the trusted proxy route/credential types to the same namespace. It never re-exports request plans, auth appliers, provider transport execution, or wire responses.
- R32. The facade runtime namespace and root exports include the runtime budget/timeouts types already required by public runtime builder signatures.
- R33. Facade contract tests depend only on facade paths when constructing provider settings, runtime budgets, and providers; the proxy milestone extends the same contract with proxy values. Direct owning-crate use remains supported and documented.
- R34. Architecture, migration, README/example, and changelog text distinguish reverse gateway from forward proxy, explain replay-limited retries, define relative-timeout start semantics, document observer privacy, and record the middleware deletion.
- R35. Provider feature flags remain additive and independently checkable. Adding the facade transport feature must not activate unrelated providers, runtime, Registry, MCP, WebSocket, or Realtime code.
- R36. No part of this work infers behavior from model/provider names, error text, environment variables, header-name substrings, prompt content, tags, JSON fences, or commercial availability. Business fallback, routing, caching, pricing, quota, compliance, and provider selection remain host-owned.

### Product Key Decisions

- **Preserve strict infrastructure authority while improving configuration ergonomics.** Governs R1-R26. *(session-settled: user-directed — chosen over simplifying by exposing raw clients, custom fetch, or broad interception that bypasses replay and endpoint controls.)*
- **Do not infer mutable product behavior or absorb host control-plane policy.** Governs R4, R6-R7, R12-R14, R30, R36. *(session-settled: user-directed — chosen over model-name/error-text heuristics and SDK-owned fallback, routing, caching, or business policy.)*
- **Treat middleware support as evidence-gated rather than a promised feature.** Governs R27-R30. *(session-settled: user-approved — chosen over committing to a middleware framework before researching whether the repository has proven consumers.)*
- **Prefer breaking cleanup to beta compatibility shells.** Governs R5, R27, R31-R34. *(session-settled: user-directed — chosen over preserving duplicated setters and an unproven Registry middleware API.)*
- **Use the AI SDK as secondary ergonomic prior art, not as Siumai's type or authority model.** Governs R8-R14, R23-R30. *(session-settled: user-directed — chosen over mechanically copying its TypeScript middleware, custom fetch, retry, and telemetry design.)*

### Key Flows

- F1. **Configure a direct provider HTTP runtime**
  - Trigger: an application builds an OpenAI, Anthropic, compatible, or other configured provider.
  - Steps: the caller constructs one `ProviderHttpTransportSettings` value; the provider combines it with its endpoint, credential, replay classifiers, and internal transport topology; branches with identical technical identity clone one transport while genuinely distinct audiences/endpoints/mechanisms remain isolated; static validation occurs before any network work.
  - Outcome: every provider exposes the same infrastructure controls without a common provider builder trait or copied setter surface, and compatible/native branches do not accidentally split connection pools, admission limits, observer state, or credential refresh.
  - Covered by: R1-R7, R23, R31-R35.

- F2. **Execute a call with a relative timeout and attempt cap**
  - Trigger: a caller supplies relative timeout and at-most attempt intent.
  - Steps: the first public call boundary resolves one deadline through an idempotent operation; every possible inner entry observes the already-resolved form; transport derives the effective attempt budget from replay safety, provider policy, and caller cap; retries, backoff, auth refresh, and stream establishment consume that same budget and deadline.
  - Outcome: the caller can narrow latency and retry exposure but cannot make an unsafe operation replayable.
  - Covered by: R8-R14, R24-R25.

- F3. **Call a provider through a trusted forward proxy**
  - Trigger: a caller explicitly selects a trusted CONNECT proxy for an HTTPS provider endpoint.
  - Steps: Siumai validates proxy and provider audiences independently; establishes the proxy connection; sends only proxy authentication during CONNECT; verifies provider TLS through the tunnel; applies provider authentication only to the tunneled request.
  - Outcome: corporate proxy use is possible without environment discovery, raw client injection, credential mixing, redirects, or retry bypass.
  - Covered by: R15-R22.

- F4. **Use remote MCP in the same enterprise network**
  - Trigger: an Agent/ToolLoop host connects to a streamable HTTP MCP endpoint through the explicit proxy route.
  - Steps: MCP reuses only the transport-owned route value; it applies that route through its own endpoint policy/client while its existing credentials, bounded JSON/SSE, lifecycle, and never-replay tool-call semantics remain authoritative.
  - Outcome: provider context and remote tool actions have explicit proxy parity without moving MCP policy into provider transport.
  - Covered by: R17, R19-R22, R36.

- F5. **Resolve a Registry model**
  - Trigger: a caller resolves `route:model` through Registry.
  - Steps: Registry resolves the immutable registration, constructs the family model, attaches canonical route context, and returns it without applying execution middleware.
  - Outcome: direct and Registry models keep the same provider behavior, and Registry no longer hides request/retry/fallback/cache wrappers.
  - Covered by: R27-R30.

- F6. **Observe an HTTP attempt loop**
  - Trigger: a provider call has a configured observer.
  - Steps: transport allocates one payload-free correlation token, emits budget resolution and attempt events, and exposes exact structural retry reasons; buffered response return, stream establishment, or a pre-establishment failure/cancellation/timeout produces the final attempt-loop event.
  - Outcome: concurrent calls can be correlated safely, and applications gain retry/handshake latency visibility without raw HTTP, response-body lifecycle claims, or model middleware authority.
  - Covered by: R14, R23-R26.

### Acceptance Examples

- AE1. Covers F1. A facade-only application starts from `ProviderHttpTransportSettings::default()`, changes one common override, and reuses that value for OpenAI and Anthropic without importing `siumai-transport`; both builders construct synchronously. The common path does not require endpoint, authentication, replay, request-plan, or transport-execution types.
- AE2. Covers R4-R5. A composite compatible provider with compatible and native branches sharing one exact endpoint/auth audience/settings uses cloned handles to one `ProviderTransport`, proving shared admission, observer, and credential refresh. Deterministic variants change exactly one dimension at a time—endpoint, credential audience, auth/signing owner, route/settings, or network mechanism—and each receives a separate transport. Old duplicated provider-level HTTP setter names are absent from production exports.
- AE3. Covers F2. Options are constructed, deliberately held before invocation, then exercised through a Siumai-owned direct trait-object call, facade helper, Registry route, and multi-step runtime call. The timer begins at each outer invocation, every inner resolve is idempotent, and the resulting absolute deadline never resets between steps. A minimal third-party fake demonstrates the documented resolution helper without implying framework enforcement.
- AE4. Covers R11-R12. Provider policy permits five attempts, caller caps two, and a semantically idempotent request executes at most twice. The same cap on `ReplaySafety::Never` executes once.
- AE5. Covers R13-R14. A response carries standard `Retry-After` beyond the configured server-delay ceiling; transport does not sleep past policy or retry early, returns the current typed failure, and emits a payload-free decline reason.
- AE6. Covers F3. Setting `HTTP_PROXY`/`HTTPS_PROXY` has no effect in Direct mode. Selecting an explicit trusted proxy produces one CONNECT exchange and the provider receives the original authenticated request through the tunnel.
- AE7. Covers R17-R19. The CONNECT fixture sees proxy authentication but no provider authorization; the provider fixture sees provider authorization but no proxy credential. Neither secret appears in `Debug`, `Display`, errors, or events.
- AE8. Covers R18, R21-R22. Proxy plus a plain local provider HTTP endpoint fails with a typed configuration error before network I/O. Provider-returned external resource download and OpenAI WebSocket/Realtime APIs do not accept or inherit provider HTTP settings, remain explicitly direct-only, and retain unchanged behavior.
- AE9. Covers F4. Streamable HTTP MCP uses the same explicit proxy route, retains bounded messages and disabled transparent replay, and never gains provider credential access.
- AE10. Covers F6. Concurrent calls expose distinct correlation tokens. A successful buffered response ends with response-returned outcome; a successful streaming handshake ends with stream-established outcome rather than completed; a network failure, cancellation, or timeout before that boundary emits one matching final attempt-loop outcome. Later body/protocol termination produces no transport-attempt event. Two configured providers share the same base observer through host wrappers and remain attributable without provider/model identity entering the transport event.
- AE11. Covers F5. Registry still resolves all six families, aliases, route-aware provider options, and route-contextualized errors after `RegistryMiddleware` is removed; `RegistryModelContext` remains public. A Registry-resolved language model runs through `Runtime::generate` and a tool loop with canonical route context intact. No middleware symbol remains outside migration text.
- AE12. Covers R31-R33. Milestone A facade compile contracts construct endpoint policy, retry policy, transport settings, observer, `RunBudget`, and `RunTimeouts` using only `siumai` paths. Milestone B extends the same namespace with the trusted proxy route and credential types without exposing execution internals.
- AE13. Covers R35. `siumai --no-default-features` remains core-only; `transport` alone activates only the transport dependency; an individual provider feature activates that provider plus transport configuration and no unrelated provider/runtime/Registry feature.
- AE14. Covers R36. Repository searches and behavior fixtures find no new model-name capability branches, error-text retry classification, environment proxy reads, arbitrary header/URL proxy callbacks, raw reqwest client injection, or business fallback/cache middleware.
- AE15. Covers R5 and R21. OpenAI retains independently named Realtime and Responses WebSocket limits/connect/session/I/O/turn controls after generic HTTP setters disappear. Alibaba video materialization retains independently named download limits/connect/download/read controls. Changing provider HTTP settings alone changes neither surface.

### Success Criteria

- Every configured provider HTTP builder has one consistent settings entry point and no duplicated provider-level HTTP settings methods.
- Callers can express relative timeout and maximum attempts without weakening replay safety or resetting deadlines across runtime steps.
- Explicit HTTPS CONNECT proxying works for provider HTTP and streamable HTTP MCP with independent credential audiences. Invalid combinations on proxy-aware HTTP surfaces fail explicitly, while WebSocket, Realtime, and provider-returned external-download APIs remain independently configured and direct-only.
- Transport observation is available across providers, explains effective retry behavior, and remains payload-free.
- The facade exposes every configuration type required by its public provider/runtime builders without exposing transport execution internals.
- Registry contains no execution middleware surface, and the repository has one fewer speculative extension system rather than one more.
- No new heuristic, global singleton, provider capability matrix, raw client escape hatch, or host business-policy feature is introduced.

### Scope Boundaries

#### Included

- `CallOptions` relative timeout and caller maximum-attempt intent.
- `ProviderHttpTransportSettings`, attempt-observer lifecycle, and retry-delay policy.
- Explicit trusted HTTPS CONNECT route for provider HTTP and streamable HTTP MCP.
- Migration of configured provider and compatibility builders to `ProviderHttpTransportSettings`.
- Curated facade transport/runtime configuration exports.
- Breaking deletion of Registry middleware and duplicated provider HTTP setters.
- Architecture, migration, changelog, examples, and deterministic offline contract tests.

#### Deferred to Follow-Up Work

- Provider WebSocket/Realtime forward proxy, SOCKS, PAC, environment proxy discovery, custom CA, mTLS, HTTP/3, arbitrary connector injection, and proxy authentication beyond bounded Basic credentials such as bearer, Negotiate, NTLM, or Kerberos.
- Proxying provider-returned resource downloads or untrusted arbitrary URLs.
- OpenTelemetry adapter, logical model/run telemetry, exporters, global registration, or default content capture.
- A first-class model middleware/layer framework, built-in default settings, caching, RAG, guardrails, fallback, simulated streaming, reasoning-tag extraction, JSON-fence cleanup, or tool-schema prompt rewriting.
- Provider credential `from_env` conveniences and automatic `.env` loading.
- New custom endpoint/replay-domain shorthand beyond making existing types reachable through the facade.
- Dynamic proxy credential refresh, proxy appliance auto-detection, and claims for named enterprise proxy products without dated deployment evidence.

#### Never Owned by Siumai Infrastructure

- Commercial route selection, fallback/load balancing, pricing, quota, compliance, account health, or regional availability.
- Model-name or provider-name capability inference.
- Error-message, prompt-content, tag, Markdown, or JSON-shape heuristics used to infer execution authority.
- Raw HTTP request/response middleware that can override endpoint, authentication, signing, replay, retry, timeout, or provider wire ownership.

### Dependencies and Sources

Internal contracts and patterns:

- `AGENTS.md`
- `docs/architecture/transport-contract.md`
- `docs/architecture/public-api.md`
- `docs/architecture/registry.md`
- `docs/adr/0010-provider-plane-and-host-control-plane.md`
- `docs/adr/0011-protocol-projection-ownership.md`
- `docs/adr/0012-provider-annotations-follow-semantic-nodes.md`
- `docs/adr/0015-validation-ownership-and-forward-compatibility.md`
- `siumai-transport/src/transport.rs`
- `siumai-transport/src/endpoint.rs`
- `siumai-transport/src/replay.rs`
- `siumai-core/src/options.rs`
- `siumai-registry/src/middleware.rs`
- `siumai-registry/src/registry.rs`
- `siumai-mcp/src/transport/http.rs`

External and secondary references verified during planning:

- Vercel AI SDK local reference checkout at `repo-ref/ai`, commit `3bc0d4f40df7` (2026-08-01), especially `packages/ai/src/middleware/wrap-language-model.ts`, `packages/provider/src/language-model-middleware/v4/language-model-v4-middleware.ts`, `packages/ai/src/registry/provider-registry.ts`, and `content/docs/03-ai-sdk-core/40-middleware.mdx`.
- AI SDK middleware documentation: https://ai-sdk.dev/docs/ai-sdk-core/middleware
- Tower `Service` and `ServiceBuilder` documentation: https://docs.rs/tower/latest/tower/
- Reqwest proxy documentation: https://docs.rs/reqwest/latest/reqwest/struct.Proxy.html
- OpenTelemetry library guidelines: https://opentelemetry.io/docs/specs/otel/library-guidelines/
- OpenTelemetry GenAI semantic conventions, currently Development and content-opt-in: https://opentelemetry.io/docs/specs/semconv/gen-ai/

---

## Planning Contract

### Key Technical Decisions

- **KTD1. Delete `RegistryMiddleware`; do not replace it in this plan.** The repository has no production middleware consumer, the existing all-family trait is wider than the evidence, direct models cannot use it, and Registry's architecture explicitly excludes execution policy. AI SDK's broad experimental hooks demonstrate ergonomics but also identity override, generate/stream substitution, and heuristic transforms that conflict with R27-R30. Ordinary Rust family-trait decorators remain the escape hatch. *(session-settled: user-approved — chosen over retaining or replacing the Registry hook because no repository consumer proves the abstraction.)*
- **KTD2. Use a transport-owned HTTP value object plus provider-author transport injection, not a provider builder trait.** A single `ProviderHttpTransportSettings` value removes duplicated storage and setters while letting each provider retain its endpoint topology, credentials, API modes, compatibility engines, retry classifier, and native resources. Compatibility engines additionally accept a prebuilt `ProviderTransport` through a narrow provider-author/internal constructor so branded providers can share one exact transport across compatible and native branches. A universal builder trait would either omit real provider differences or become a capability matrix, while independently rebuilding every internal transport would split pools and control state. *(session-settled: user-approved — chosen over a universal builder trait or independently rebuilt transports because both create a second source of truth.)*
- **KTD3. Keep endpoint destination and forward proxy as separate typed authorities.** Endpoint configuration continues to define provider origin and credential audience. The explicit trusted CONNECT route defines how traffic reaches that origin and records the DNS/peer-validation trust transfer. A custom base URL remains a reverse gateway, not a proxy alias. *(session-settled: user-directed — chosen over heuristic endpoint/proxy inference or raw-client injection to preserve authority boundaries.)*
- **KTD4. Resolve relative timeout exactly once at logical-call entry.** Starting at options construction surprises delayed callers; resolving independently in transport or every runtime step extends total time. The outer entry resolves to an absolute deadline, and all inner layers propagate it. *(session-settled: user-approved — chosen over construction-time or per-layer timers because those consume time early or reset the budget.)*
- **KTD5. Caller retry controls only narrow provider policy.** The effective attempt count is derived from replay proof, provider policy, and caller cap. Cloneability, status code, lack of response bytes, or absence of a stream event never proves replay safety. *(session-settled: user-directed — chosen over heuristic retry permission because caller ergonomics must not expand replay authority.)*
- **KTD6. Treat observation as attempt telemetry, not model middleware or protocol completion.** Transport can truthfully report its own attempt loop and stream establishment but cannot claim semantic model completion. Logical model/run telemetry and OpenTelemetry mapping remain separate future adapters. *(session-settled: user-approved — chosen over a model telemetry framework because transport lacks protocol/run completion ownership.)*
- **KTD7. Support a deliberately narrow proxy v1.** HTTPS CONNECT for provider HTTP and MCP HTTP is the smallest slice that serves a real enterprise-network need while keeping credential and replay boundaries testable. WebSocket, Realtime, provider-returned external resource downloads, SOCKS, PAC, custom TLS, and raw client injection remain deferred. *(session-settled: user-approved — chosen over broad proxy/client customization because the narrow route is testable without becoming a network policy platform.)*
- **KTD8. Delete old provider HTTP setters after one migration window inside the branch.** The implementation may temporarily carry old and new setters while packages migrate, but the landed public surface contains only the unified settings entry point plus genuinely lifecycle-specific session APIs. Beta compatibility aliases would preserve the duplication this refactor exists to remove. *(session-settled: user-directed — chosen over compatibility aliases because breaking cleanup is explicitly allowed.)*
- **KTD9. Add a curated facade configuration namespace instead of re-exporting transport execution.** Facade users must be able to name public builder parameter types, but `ProviderTransport`, request plans, auth appliers, and raw responses remain owning-crate APIs. *(session-settled: user-approved — chosen over broad re-exports because facade ergonomics does not require execution authority.)*
- **KTD10. Keep observer attribution host-owned.** The transport observer reports only one bounded attempt loop and an opaque correlation token. Applications that need provider, route, account, or tenant attribution wrap the observer when configuring each provider; transport does not accept or synthesize those identities. *(session-settled: user-approved — chosen over provider-aware transport events because attribution context belongs to the installing host.)*
- **KTD11. Preserve non-HTTP lifecycle controls explicitly.** Removing generic provider HTTP setters must not remove existing WebSocket, Realtime, or external-download controls. OpenAI receives dedicated Realtime and Responses WebSocket limits/connect controls alongside its existing session/I/O/turn controls, and Alibaba video materialization receives dedicated download controls; these values never inherit the HTTP route. *(session-settled: user-approved — chosen over silently inheriting or dropping controls because those lifecycles are not stateless provider HTTP.)*

### High-Level Technical Design

The diagram is an ownership map, not an implementation class diagram:

```text
Host application
  ├─ explicit provider / Registry choice
  ├─ CallOptions (deadline, cancellation, caller attempt cap)
  ├─ ProviderHttpTransportSettings
  │    ├─ limits
  │    ├─ provider retry policy
  │    ├─ HTTP timeouts
  │    ├─ payload-free observer
  │    └─ HttpTransportRoute
  │         ├─ Direct (default; environment proxy disabled)
  │         └─ Trusted CONNECT proxy
  └─ optional ordinary Rust model wrapper (host code, no Siumai middleware registry)

Configured provider / compatibility engine
  ├─ endpoint and credential audience
  ├─ auth/signing
  ├─ API mode and codec
  ├─ shared ProviderTransport handle for identical technical identity
  ├─ operation ReplaySafety
  └─ provider retry classification
          │
          ▼
siumai-transport protected HTTP kernel
  ├─ resolve one absolute deadline
  ├─ derive effective attempts
  ├─ Direct DNS proof OR explicit proxy trust delegation
  ├─ credential phase separation
  ├─ replay-safe attempt loop
  ├─ body/header/queue bounds
  └─ sanitized attempt events
          │
          ├─ buffered HTTP response
          └─ established byte stream → protocol-owned decoder/terminal

Registry remains parallel to configuration:
route:model → registration → family model → route-context wrapper → caller
```

The implementation dependency graph is deliberately not the numeric order of every optional unit:

```text
U1 Registry cleanup ───────────────────────────────┐
                                                  ├─→ U9 Direct closure
U2 call intent → U3 HTTP settings → U6 → U7 → U8 ┘
                                  │            │
                                  └────────────┴─→ U4 CONNECT → U5 MCP route → U10 proxy closure
```

The proxy connection is a two-audience sequence, not one merged request:

```text
Host
  └─ TLS to validated proxy + optional ProxyBasicCredential
       └─ CONNECT logical-provider-host:443
            └─ provider TLS hostname/certificate verification
                 └─ provider HTTP request + provider credential
```

The proxy connection therefore has two security phases:

1. connect/authenticate to the validated proxy audience;
2. establish a CONNECT tunnel to the logical HTTPS destination, verify destination TLS, then apply provider authentication only inside the tunnel.

The implementation must not flatten those phases into one header map or one credential callback.

### Sequencing

1. Remove the speculative Registry middleware surface independently; route-context behavior remains green throughout.
2. Add relative timeout and caller retry cap to core, then teach transport to derive one effective deadline/attempt budget.
3. Introduce `ProviderHttpTransportSettings` and precise attempt observer events while existing provider setters still compile.
4. Migrate compatibility engines and flagship providers, then remaining providers, while Direct remains the only route.
5. Repair facade reachability after U6-U7 have deleted duplicated HTTP setters and lifecycle-specific non-HTTP replacements are green.
6. Add the explicit proxy route to provider stateless HTTP transport as a separately mergeable extension.
7. Adapt streamable HTTP MCP to the route type without sharing provider settings, auth, or retry policy.
8. Close architecture, migration, examples, and final deletion/feature gates.

U1 is independent. U2 precedes U3. U6 depends only on U3, U7 depends on U6, and U8 depends on U6-U7. Together U1-U3 and U6-U8 form the Direct infrastructure milestone and may merge without proxy support. U4 depends on U3 and U8 so the route and its facade reachability land together; U5 depends on U4 but is independent of provider migration because providers already consume the settings value. U9 closes the Direct milestone. U10 extends that closure after U4-U5 and completes the proxy milestone.

### Delivery Milestones

- **Milestone A — Direct infrastructure ergonomics:** U1-U3 and U6-U9. This milestone is independently releasable and must not wait for proxy demand, proxy appliance access, or MCP routing work.
- **Milestone B — Explicit CONNECT routing:** U4-U5 and U10. This milestone adds only the bounded route contract and can be delayed or reverted without reopening provider builder, timeout, retry, observer, Registry, or facade architecture.

### System-Wide Impact

- **Provider builders:** public method deletion across provider and compatibility crates. Endpoint/credential/provider-native configuration remains provider-owned. Composite builders become the explicit owner of transport sharing: exact technical identity shares one cloneable transport, while distinct audiences/endpoints/mechanisms remain isolated.
- **Facade features:** provider features gain a narrow transport configuration dependency; core-only and no-default builds must remain lean.
- **Runtime and agents:** relative deadlines must not reset across model steps. MCP gains explicit route parity but no retry/fallback behavior.
- **Server projection:** `ServerGateway` continues to forward the updated `CallOptions` and runtime semantics without starting a second deadline or losing the caller attempt cap.
- **Registry:** one public extension trait and related snapshot/builder inspection APIs disappear. Route identity and exact provider-option targeting remain unchanged.
- **Transport security:** explicit proxy mode changes who resolves and connects to destination addresses. This is a named trust transfer, not a relaxation of Direct mode.
- **Credentials:** proxy and provider credentials acquire separate typed owners, audiences, merge phases, and redaction fixtures.
- **Streaming:** transport events stop calling a successfully established stream `Completed`; protocol terminal semantics remain untouched.
- **Observability:** applications can install one sanitized observer across providers. Existing OpenAI-only code migrates without gaining payload access.
- **Documentation and ecosystem:** beta users receive a breaking migration table. No source parser or compatibility alias is added to preserve old setter spellings.

### Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Proxy implementation accidentally sends provider auth during CONNECT | Build proxy and provider header sets in separate phases; use dual-server canary fixtures and sentinel scans. |
| Trusted proxy mode is described as preserving local DNS proof | State the authority transfer in type/rustdoc/architecture; Direct remains the only locally validated DNS path. |
| Provider builder migration silently omits an internal engine | Test composite providers with distinct local endpoints/observers; compatibility engines migrate before branded wrappers. |
| Transport sharing crosses an authentication or endpoint boundary | Define an exact sharing key from endpoint, credential audience/auth owner, route, and settings; test both same-key sharing and distinct-key isolation. Never infer equivalence from provider/model labels. |
| Relative timeout is resolved more than once | Add an explicit unresolved/resolved internal state and delayed-construction plus multi-step tests. |
| Caller cap is mistaken for retry permission | Derive effective attempts only after replay proof; keep `Never` at one attempt in every fixture. |
| Observer events leak identifiers or misstate stream completion | Keep a closed payload-free event schema and separate stream-established from protocol terminal. |
| A shared observer cannot attribute events to a configured provider | Keep transport events identity-free; prove the host-owned wrapper pattern with two providers and one shared sink. |
| Removing Registry middleware breaks a hidden user | Record the beta break clearly and show ordinary wrapper/explicit post-resolution composition as the replacement pattern; do not ship a compatibility trait. |
| Facade transport feature pulls unrelated crates | Add no-default and per-feature metadata/check gates; re-export configuration types only. |
| Proxy support creates pressure for raw clients, PAC, or TLS customization | Keep R22 and KTD7 explicit; reject unsupported route/surface combinations before network I/O. |
| Proxy credentials rotate while a configured client is long-lived | Treat credentials as immutable snapshots and require rebuilding the configured provider/MCP client; do not add a callback or global refresh owner. |
| Deleting generic HTTP setters silently changes WebSocket/Realtime/download behavior | Add lifecycle-specific replacement controls and regression fixtures before deleting the old setters. |

---

## Implementation Units

### U1. Remove speculative Registry middleware

**Goal:** restore Registry to deterministic lookup plus route-context projection and delete the unproven execution-decoration surface.

**Requirements:** R27-R30, R34, R36; F5; AE11, AE14.

**Dependencies:** None.

**Files:**

- `siumai-registry/src/middleware.rs`
- `siumai-registry/src/registry.rs`
- `siumai-registry/src/error.rs`
- `siumai-registry/src/lib.rs`
- `siumai-registry/tests/` and in-module Registry tests
- `siumai/src/registry.rs`
- `siumai/src/prelude.rs`
- `siumai/tests/facade_contract.rs`
- `docs/architecture/registry.md`
- `docs/architecture/public-api.md`
- `docs/migration/siumai-next.md`
- `CHANGELOG.md`

**Approach:**

- Delete `RegistryMiddleware`, `MiddlewareStack`, builder/snapshot middleware methods, and public exports.
- Preserve the public `RegistryModelContext` used by typed resolve errors; move lookup contextualization and the six route family wrappers into a route-owned private module with no middleware terminology.
- Resolve each family directly, then apply only the canonical route wrapper.
- Preserve route-aware error context, route-bound provider-option selection, alias resolution, all six family constructors, and synchronous/network-free behavior.
- Document that an ordinary host decorator must forward the complete descriptor and `route_id()`; this is guidance, not a new Siumai decorator API.
- Remove middleware counting/order/identity-drift tests rather than replacing them with another hook test.

**Test scenarios:**

- Direct and aliased routes resolve all six families and expose the canonical route.
- Required exact-target provider options still cross the route wrapper and reach the configured provider model.
- Provider construction errors and direct/stream errors retain route context.
- A Registry-resolved language model passes through `Runtime::generate` and a tool loop while retaining route defaults and exact-target provider option selection.
- Public/facade compile contracts no longer import any Registry middleware symbol.
- Repository symbol search finds middleware names only in migration/history text permitted by the plan.

**Verification:** `siumai-registry` nextest/clippy, facade Registry feature contract, scoped rustdoc/doctest, and deletion search are green.

### U2. Add one-shot relative deadlines and caller attempt caps

**Goal:** make common per-call timing/retry intent ergonomic while preserving one deadline and replay-first authority.

**Requirements:** R8-R14, R36; F2; AE3-AE5.

**Dependencies:** None.

**Files:**

- `siumai-core/src/options.rs`
- `siumai-core/src/lib.rs`
- `siumai-core/tests/public_contract_compile.rs`
- `siumai/src/families.rs`
- `siumai-registry/src/registry.rs`
- focused Registry deadline contract tests
- `siumai-transport/src/transport.rs`
- `siumai-transport/src/replay.rs`
- `siumai-transport/src/error.rs`
- `siumai-transport/tests/transport_contract.rs`
- `siumai-runtime/src/call.rs`
- `siumai-runtime/src/single_step.rs`
- `siumai-runtime/src/engine.rs`
- `siumai-runtime/src/durable.rs`
- `siumai-runtime/src/structured_run.rs`
- `siumai-server/src/gateway.rs`
- focused server gateway call/runtime contract tests
- direct runtime timeout contract tests

**Approach:**

- Replace the binary retry intent with provider-policy-or-at-most semantics; keep a convenience path for one attempt without a separate authority model.
- Add unresolved relative timeout state and one explicit resolution operation that merges with an absolute deadline and removes the relative value.
- Resolve at facade family helpers, Registry route entry, and high-level runtime entry; require every Siumai-owned configured provider/compatibility family-model implementation to invoke the same idempotent operation as the direct-trait-call safeguard during U6-U7 migration. Expose that operation as the documented third-party implementation seam. Inner transport only receives/respects the resolved deadline.
- Resolve structured-output options before computing the runner's shared deadline or cloning options for a repair attempt, so initial and repair calls consume one absolute deadline.
- Keep the scope to model-family calls that accept `CallOptions`; do not invent a common timeout carrier for provider-native resource/session APIs with different lifecycles.
- Derive effective attempts from replay proof, provider maximum, and caller cap in that order.
- Define explicit policy for server retry advice above the maximum allowed delay or remaining deadline; return the current failure instead of retrying early.

**Test scenarios:**

- Construction delay does not consume a relative timeout before invocation.
- Direct trait-object, facade helper, Registry, and runtime entry paths resolve identically; absolute plus relative timeout uses the earliest deadline.
- Runtime multi-step, durable, Registry, auth refresh, queue wait, and backoff paths do not reset the deadline.
- Structured-output initial and repair attempts share the same resolved deadline and cannot restart a relative timeout.
- `ServerGateway` single-call and tool-loop entries preserve resolved relative timeout and caller attempt-cap semantics when forwarding into Runtime.
- Provider policy 5 plus caller cap 2 produces at most two attempts for replay-safe requests and one for `ReplaySafety::Never`.
- Cancellation during queue/backoff remains cancellation, and zero/overflow duration fails before network submission.
- Oversized `Retry-After` is declined structurally without sleeping past policy or retrying sooner.

**Verification:** core, transport, runtime, Registry, and server focused suites pass; `ServerGateway` contracts and public compile tests demonstrate the new call options; no retry path bypasses `ReplaySafety`.

### U3. Introduce provider HTTP transport settings and precise attempt events

**Goal:** create the single configuration carrier that provider builders will consume and make attempt observation truthful and complete.

**Requirements:** R1-R5, R7, R14, R23-R26, R36; F1, F6; AE1-AE2, AE10.

**Dependencies:** U2.

**Files:**

- `siumai-transport/src/settings.rs` (new)
- `siumai-transport/src/transport.rs`
- `siumai-transport/src/replay.rs`
- `siumai-transport/src/error.rs`
- `siumai-transport/src/lib.rs`
- `siumai-transport/tests/transport_contract.rs`

**Approach:**

- Add one immutable/clonable `ProviderHttpTransportSettings` value with validated limits, retry policy, timeouts, observer, and Direct route default.
- Name the public entry `with_http_transport_settings(...)`; remove any implication that the value configures provider WebSocket, Realtime, external downloads, or MCP sessions.
- Keep endpoint, auth, resolver, provider retry classifier, and request replay declaration as explicit `ProviderTransportBuilder` inputs outside the value.
- Keep `ProviderTransport` cheaply cloneable as the shared pool/admission/observer/credential-refresh handle; do not place endpoint/auth ownership inside the settings value merely to make sharing convenient.
- Add one settings application path to the transport builder and reduce its individual infrastructure setters to private assembly helpers where still needed internally.
- Replace ambiguous `Completed` observation with an attempt-loop outcome that distinguishes buffered response return, stream establishment, failure, cancellation, and timeout.
- Allocate one opaque `TransportCallId` per transport execution and include it in every event so a shared observer can correlate concurrent attempt loops without receiving user/provider identifiers.
- Emit the resolved attempt budget and limiting authority before the first attempt.
- Keep events structurally bounded and hand-write redacted `Debug` for settings and any new credential-bearing placeholders.

**Test scenarios:**

- Default settings exactly preserve current Direct-mode limits, retry, and timeout behavior.
- Invalid limits/timeouts fail before DNS/client construction.
- Buffered, streaming, retried, declined, failed, cancelled, and timed-out calls each produce a deterministic event sequence and one final attempt-loop outcome.
- Response-head events expose only the structural retry fields named by R14; raw header collections, names, and values never enter the observer API or diagnostics.
- Concurrent executions have distinct call IDs, stable IDs across their own retries, and no body/protocol terminal event after response return or stream establishment.
- Two provider-scoped observer wrappers feed one shared sink and preserve host attribution without adding provider/model/route fields to `TransportEvent`.
- Observer event `Debug` and sentinel tests contain no URL, header, body, prompt, or credential data.
- Settings cloning shares the observer intentionally while preserving immutable values.
- Cloning one built `ProviderTransport` shares pool/admission/observer state, while independently built transports remain isolated even when their human-readable provider label is identical.

**Verification:** transport nextest/clippy and rustdoc are green; existing replay/bounds fixtures remain unchanged except for intentional event names/semantics.

### U4. Add an explicit trusted CONNECT route to provider HTTP transport

**Goal:** support a bounded provider-HTTP forward-proxy slice without raw network injection or credential/replay authority loss.

**Requirements:** R15-R19, R21-R22, R25-R26, R31-R33, R36; F3; AE6-AE8, AE12, AE14.

**Dependencies:** U3, U8.

**Files:**

- `siumai-transport/src/proxy.rs` (new)
- `siumai-transport/src/settings.rs`
- `siumai-transport/src/transport.rs`
- `siumai-transport/src/endpoint.rs`
- `siumai-transport/src/error.rs`
- `siumai-transport/src/lib.rs`
- `siumai-transport/tests/transport_contract.rs`
- `siumai/src/transport.rs`
- `siumai/tests/facade_contract.rs`

**Approach:**

- Model `Direct` and explicit trusted CONNECT as a closed transport route.
- Accept only a proxy origin with no path, query, fragment, or userinfo. Permit a public `https://` proxy or an explicitly granted `LocalExplicit` proxy; reject an arbitrary cleartext public proxy.
- Validate the proxy endpoint/audience independently. Expose only bounded `ProxyBasicCredential` construction and apply it through reqwest's proxy Basic-auth path; do not accept raw authorization headers or alternative authentication schemes in v1.
- Treat proxy credentials as an immutable route snapshot. Document rebuilding the provider/MCP client as the rotation mechanism and reject any callback/global-refresh extension in this unit.
- Reject proxy credentials on a cleartext `LocalExplicit` proxy before client construction; cleartext local proxying is permitted only without authentication.
- Configure reqwest with one explicit proxy only; continue disabling environment proxies, redirects, referer, and reqwest retry.
- Use a route-specific resolver/peer validator: Direct resolves and validates the provider origin, while proxy mode resolves and validates the proxy endpoint/peer. Do not pass proxy peers through provider-origin `validate_remote` checks.
- Restrict v1 provider destinations to HTTPS, reject local/plain destinations, and document that the proxy owns destination DNS/peer routing while Siumai retains logical destination/TLS verification.
- Apply proxy authentication only to proxy negotiation and provider authentication only to the tunneled provider request.
- Scope `ProviderHttpTransportSettings` to stateless HTTP APIs. WebSocket, Realtime, and external resource downloader configuration remains independent and direct-only; no proxy state is propagated into those APIs merely to reject it.
- Extend the already-curated facade transport namespace with only the trusted route, endpoint, and proxy-credential configuration types.

**Test scenarios:**

- Direct mode ignores process proxy environment variables.
- Explicit proxy performs one CONNECT, preserves target host/TLS, and does not auto-follow redirects or retry outside Siumai policy.
- Proxy DNS and peer validation targets the proxy endpoint, while the tunneled origin retains URL, TLS hostname/certificate, credential-audience, redirect, replay, and bounds checks.
- Dual sentinels prove proxy/provider credential phase separation and diagnostic redaction.
- Configuring a credential on a cleartext local proxy fails with a typed pre-network error, while the same credential is accepted for an HTTPS proxy.
- Invalid/oversized Basic username or password input fails before client construction, and no username/password appears in diagnostics. Opaque authorization, bearer, Negotiate, and URL-userinfo paths are absent from the API.
- HTTP 407, CONNECT failure, authentication failure, cancellation, deadline, queue bounds, and oversized response paths retain typed sanitized errors.
- Local/plain provider targets reject the proxy route before network I/O; independently configured provider-returned external resource downloads, WebSocket, and Realtime APIs remain unchanged and do not inherit the HTTP route.
- Documentation and type names claim bounded HTTPS CONNECT support only; no fixture or rustdoc claims certification for a named corporate proxy product.
- Facade-only construction of the trusted route uses no owning-crate or low-level execution imports.

**Verification:** transport full-feature nextest/clippy plus facade contract, doctest, clippy, and no-default `transport` feature checks pass; proxy fixtures are offline and bounded; no environment lookup or raw client API is public.

### U5. Add the HTTP route adapter to streamable HTTP MCP

**Goal:** give remote MCP the same explicit enterprise route choice without importing provider transport settings, provider authentication, or retry semantics.

**Requirements:** R17, R19-R20, R22, R25-R26, R36; F4; AE7, AE9, AE14.

**Dependencies:** U4.

**Files:**

- `siumai-mcp/src/config.rs`
- `siumai-mcp/src/client.rs`
- `siumai-mcp/src/transport/http.rs`
- `siumai-mcp/src/lib.rs`
- MCP HTTP contract tests

**Approach:**

- Add `HttpTransportRoute` to the streamable HTTP MCP configuration while retaining `McpHttpEndpointPolicy`, MCP bearer authentication, body/SSE limits, session lifecycle, and error ownership.
- Apply the same route consistently to MCP POST, GET/SSE, and DELETE operations through MCP's own bounded client.
- Keep proxy and MCP-origin credentials in separate phases; never expose provider credentials or accept `ProviderHttpTransportSettings`/`ProviderTransport` in MCP.
- Preserve MCP's disabled transparent replay and side-effecting `tools/call` semantics; the route changes network reachability only.
- Keep stdio MCP unchanged and do not claim that MCP endpoint/DNS policy is identical to provider transport policy.

**Test scenarios:**

- Direct MCP remains environment-proxy independent.
- POST, GET/SSE, and DELETE all use the selected explicit route and retain existing bounds and session behavior.
- Proxy authentication appears only during CONNECT; MCP bearer authentication appears only in tunneled origin requests; diagnostics redact both.
- Proxy failure/cancellation/deadline remains typed and no `tools/call` is replayed or converted into a provider retry.
- Stdio and provider transport types remain absent from the new MCP HTTP configuration surface.

**Verification:** MCP all-feature nextest/clippy and bounded local CONNECT/SSE fixtures pass serially; no live MCP server or credentials are used.

### U6. Migrate compatibility engines and flagship providers

**Goal:** prove the settings contract across the shared engines and the largest provider topologies before migrating the rest of the workspace.

**Requirements:** R1-R9, R21, R23-R25, R35-R36; F1-F2, F6; AE1-AE3, AE10, AE15.

**Dependencies:** U3.

**Files:**

- `siumai-openai-compatible/src/configured/provider.rs`
- `siumai-openai-compatible/src/configured/model.rs`
- `siumai-openai-compatible/src/configured/execution.rs`
- `siumai-anthropic-compatible/src/provider.rs`
- `siumai-anthropic-compatible/src/model.rs`
- `siumai-provider-openai/src/configured/provider.rs`
- `siumai-provider-openai/src/configured/model.rs`
- `siumai-provider-openai/src/configured/embedding.rs`
- `siumai-provider-openai/src/configured/image.rs`
- `siumai-provider-openai/src/configured/speech.rs`
- `siumai-provider-openai/src/configured/transcription.rs`
- `siumai-provider-anthropic/src/provider.rs`
- `siumai-provider-gemini/src/provider.rs`
- `siumai-provider-gemini/src/language.rs`
- `siumai-provider-gemini/src/generate_content.rs`
- `siumai-provider-gemini/src/embedding.rs`
- `siumai-provider-gemini/src/image.rs`
- `siumai-provider-gemini/src/speech.rs`
- `siumai-provider-google-vertex/src/providers/anthropic_vertex/provider.rs`
- direct provider/compatibility contract tests

**Approach:**

- Replace duplicated HTTP fields with one settings snapshot in each builder/runtime.
- Add a narrow provider-author/internal compatibility-engine constructor that accepts a prebuilt `ProviderTransport`; do not expose raw client, auth, endpoint, or request-plan mutation through this seam.
- At each branded provider build, create one transport per exact technical identity tuple (endpoint, credential audience/auth owner, route, settings) and clone it into compatible language and native resource branches that share that tuple.
- Build separate transports for different Vertex regions/projects, resource origins, signing audiences, or HTTP versus WebSocket/Realtime mechanisms even when they belong to one branded provider.
- Apply settings to stateless HTTP execution and native resource clients; keep provider-owned session/Realtime timeout and limit configuration separate.
- Resolve relative call timeouts idempotently at every Siumai-owned implementation of the six public model-family traits before validation, request planning, or transport submission; direct trait calls must not start the timer later inside transport.
- Ensure OpenAI observer behavior is no longer special and migrate its existing tests to the common event contract.
- Before deleting OpenAI's generic HTTP limits/connect setters, add dedicated provider-level Realtime limits/connect controls and Responses WebSocket limits/connect controls; retain the existing session/I/O/turn controls. HTTP settings must not feed either session runtime.
- Preserve compatibility profile/dialect ownership, provider-specific retry classifiers, endpoints, credentials, replay domains, exact provider-instance option targeting, and native resources.
- Keep old setters only while this unit is in flight; do not expose aliases in the completed unit.

**Test scenarios:**

- One `ProviderHttpTransportSettings` value configures OpenAI Chat/Responses/resources and Anthropic Messages/resources without changing request wire or replay safety.
- OpenAI and Gemini language, embedding, image, speech, and transcription family entries all resolve delayed relative options once before planning and preserve the same absolute deadline through retries/stream establishment.
- OpenAI-compatible and Anthropic-compatible engines receive settings through their supported builders and custom endpoints remain explicit.
- A branded provider with compatible and native branches sharing one technical identity proves that admission permits, observer events, and credential refresh are shared. Separate fixtures change endpoint, credential audience, auth/signing owner, route/settings, and HTTP-versus-session mechanism one at a time and prove isolation for every key dimension.
- Gemini and Vertex continue their provider-specific endpoint/auth/signing behavior while sharing transport settings.
- Observer sequences match the common transport contract for at least two providers and one compatibility engine.
- OpenAI Responses WebSocket/Realtime exposes no `ProviderHttpTransportSettings` coupling, retains independently configurable limits/connect/session/I/O/turn behavior, and remains green under direct configuration.

**Verification:** all six provider/compatibility package suites, no-default feature checks, clippy, and representative facade feature checks pass serially.

### U7. Migrate remaining configured provider builders and delete duplicate setters

**Goal:** finish workspace-wide settings adoption without introducing a common provider builder abstraction.

**Requirements:** R1-R9, R21, R23, R35-R36; F1-F2; AE1-AE3, AE14-AE15.

**Dependencies:** U6.

**Files:**

- `siumai-provider-alibaba/src/provider.rs`
- `siumai-provider-alibaba/src/embedding.rs`
- `siumai-provider-cohere/src/configured/provider.rs`
- `siumai-provider-cohere/src/configured/model.rs`
- `siumai-provider-deepgram/src/provider.rs`
- `siumai-provider-deepgram/src/model.rs`
- `siumai-provider-deepgram/src/speech.rs`
- `siumai-provider-deepseek/src/provider.rs`
- `siumai-provider-elevenlabs/src/configured/provider.rs`
- `siumai-provider-elevenlabs/src/configured/model.rs`
- `siumai-provider-elevenlabs/src/configured/transcription.rs`
- `siumai-provider-groq/src/provider.rs`
- `siumai-provider-groq/src/speech.rs`
- `siumai-provider-groq/src/transcription.rs`
- `siumai-provider-minimax/src/provider.rs`
- `siumai-provider-minimax/src/portable.rs`
- `siumai-provider-moonshotai/src/provider.rs`
- `siumai-provider-volcengine/src/provider.rs`
- `siumai-provider-volcengine/src/image.rs`
- `siumai-provider-xai/src/providers/xai/provider.rs`
- `siumai-provider-xai/src/providers/xai/media.rs`
- affected provider contract tests and examples

**Approach:**

- Replace repeated fields/private settings structs with the transport-owned value.
- Propagate one snapshot to every internal HTTP runtime in composite providers; share a built transport only when endpoint, credential audience, auth/signing owner, HTTP route/settings, and network mechanism all match, and keep provider-specific endpoint/replay/profile decisions explicit.
- Resolve relative call timeouts idempotently at every remaining Siumai-owned language, embedding, rerank, image, speech, and transcription entry before validation, request planning, or transport submission.
- Add observer parity where it was previously unavailable and retry settings where the transport already supports them.
- Delete provider-level HTTP setters with inconsistent names, including `with_limits` versus `with_transport_limits`; retain only lifecycle-specific non-HTTP controls.
- Before deleting Alibaba's generic transport limits/timeouts, add explicit video-materialization download limits/connect/download/read controls and prove that provider HTTP settings do not alter the `ResourceDownloader`.
- Do not add a macro, build script, generic provider trait, or textual API gate to keep the list synchronized.

**Test scenarios:**

- Each provider builder accepts `ProviderHttpTransportSettings` and preserves its default endpoint/auth/profile behavior.
- A trait-implementation audit covers all six public model families; delayed relative options resolve once at each direct entry, and options cloned or forwarded internally retain the absolute form.
- Composite providers apply identical configured limits/timeouts/observer/route to all stateless HTTP branches and share one transport for same-key branches. One-dimension fixtures prove isolation across endpoint, credential audience, auth/signing owner, route/settings, and network mechanism.
- One observer fixture from a native provider and one branded compatible provider prove parity.
- Provider-specific resources/jobs remain bounded and do not silently drop settings.
- Alibaba video materialization preserves its existing bounds and timeout configurability under the dedicated download controls and remains direct-only.
- Production search finds no deleted duplicated provider HTTP setter declarations outside migration text and session-specific APIs.

**Verification:** affected provider package nextest/clippy runs pass serially; facade all-provider feature check and cargo metadata show no new dependency cycle.

### U8. Repair facade configuration reachability

**Goal:** let facade-only applications name and construct every infrastructure type required by public provider/runtime builders.

**Requirements:** R31-R35; F1; AE1, AE12-AE13.

**Dependencies:** U6, U7.

**Files:**

- `siumai/Cargo.toml`
- `siumai/src/lib.rs`
- `siumai/src/prelude.rs`
- `siumai/src/transport.rs` (new)
- `siumai/src/runtime.rs`
- `siumai/tests/facade_contract.rs`
- `siumai/examples/openai_flagship.rs`
- `siumai/examples/anthropic_flagship.rs`
- `siumai/README.md`
- root `README.md`

**Approach:**

- Add a narrow optional facade transport dependency/feature and make each provider feature activate it.
- Re-export the Direct-milestone configuration and observation types; keep execution/auth/request-plan types out of the facade. U4 later adds only the proxy configuration types to this namespace.
- Export runtime budget/builder/timeouts/error types already required by public runtime methods.
- Rewrite facade contracts so they never import `siumai_transport` or `siumai_runtime` directly.
- Keep no-default facade core-only and avoid activating MCP, WebSocket, Realtime, Registry, or runtime from the transport feature.

**Test scenarios:**

- Facade-only compile contract builds provider settings, observer, endpoint, retry cap, relative timeout, and runtime budget; U4 owns the additive facade proxy fixture.
- `--no-default-features`, `transport`, individual provider, Registry, runtime, Responses WebSocket, and Realtime feature combinations remain independently checkable.
- Examples compile against only documented facade paths and required features.
- Curated namespace contains no `ProviderTransport`, auth applier, request plan, raw response, or low-level socket type.

**Verification:** facade nextest/doctest/clippy and feature checks pass; cargo metadata confirms the intended optional dependency graph.

### U9. Close the Direct infrastructure milestone

**Goal:** make Milestone A independently releasable, document proxy routing as deferred, and remove Direct-migration transitional code/tests.

**Requirements:** R1-R14, R21, R23-R36 except the Milestone B proxy additions in R31-R34; F1-F2, F5-F6; AE1-AE5, AE10-AE15.

**Dependencies:** U1-U3, U6-U8.

**Files:**

- `docs/architecture/transport-contract.md`
- `docs/architecture/public-api.md`
- `docs/architecture/registry.md`
- `docs/architecture/overview.md`
- `docs/migration/siumai-next.md`
- `CHANGELOG.md`
- provider/Registry/transport tests and examples made stale by Milestone A

**Approach:**

- Document the Direct settings owner, deadline semantics, retry cap, replay-limited attempts, observer lifecycle, host-owned observer attribution, and independent non-HTTP lifecycle controls.
- Record a concise old-to-new provider builder migration table and Registry middleware deletion guidance.
- Explain that ordinary model decorators remain host code and that no middleware framework, fallback/cache engine, raw client hook, or OTel integration was added.
- State that forward proxy and MCP route support remain deferred; do not publish partial proxy types or claims in a Milestone A release.
- Delete transitional aliases, duplicate settings structs, obsolete observer tests, and stale claims in the same change that makes them unnecessary.
- Keep automation thin: use compiler/feature/tests and bounded fixtures rather than adding policy scripts or source parsers.

**Test scenarios:**

- Migration examples compile against the Milestone A public API.
- Documentation consistently distinguishes Direct transport from custom reverse-gateway endpoints and marks forward proxying as deferred.
- Searches find no stale public Registry middleware or duplicated provider HTTP setter names outside migration notes, no environment proxy reads, and no raw client/custom fetch extension.
- Security sentinel scan covers provider credentials, endpoint query, response body, prompt, and tool payload.

**Verification:** docs/doctests, workspace formatting, diff checks, Milestone A symbol searches, and the Milestone A integration matrix are green.

### U10. Close the explicit CONNECT milestone

**Goal:** extend the accepted Direct architecture with the bounded provider/MCP CONNECT route and complete the full-plan documentation and release contract.

**Requirements:** R15-R22, the Milestone B proxy additions in R31-R34, R36; F3-F4; AE6-AE9, AE12, AE14.

**Dependencies:** U4-U5, U9.

**Files:**

- `docs/architecture/transport-contract.md`
- `docs/architecture/public-api.md`
- `docs/architecture/overview.md`
- `docs/migration/siumai-next.md`
- `docs/providers/support-policy.md` only if a proxy support claim is added
- `siumai-mcp/README.md`
- facade/provider examples that demonstrate the trusted route without live credentials
- `CHANGELOG.md`

**Approach:**

- Document Direct versus trusted CONNECT, proxy/provider credential phases, immutable Basic credential rotation, destination-DNS trust transfer, MCP route reuse, and every explicitly unsupported proxy surface.
- Extend the migration table and facade examples with the proxy types without changing the already-landed Direct settings path.
- Describe v1 as bounded HTTPS CONNECT support, not certification for a named enterprise proxy product or environment.
- Delete abandoned proxy experiments and stale Direct-only claims; do not add live canaries, environment discovery, policy scripts, or proxy appliance matrices.

**Test scenarios:**

- Migration and facade examples compile against the complete public API.
- Documentation consistently uses reverse gateway, forward proxy, Direct, trusted proxy, proxy audience, and provider audience with their exact meanings.
- Searches find no environment proxy reads, raw proxy/client/fetch extension, URL userinfo credential path, or unsupported authentication scheme.
- Security sentinel scan covers proxy credentials, provider credentials, endpoint query, response body, prompt, and tool payload.

**Verification:** docs/doctests, workspace formatting, diff checks, final proxy/middleware/setter searches, and the full integration matrix are green.

---

## Verification Contract

### Per-Unit Gates

- U1: `siumai-registry` full-feature nextest and clippy; facade Registry contract; Registry rustdoc/doctest; middleware deletion search.
- U2: `siumai-core`, `siumai-transport`, `siumai-runtime`, and `siumai-server` nextest/clippy; direct delayed-timeout, structured repair, server gateway, multi-step deadline, replay safety, caller cap, Retry-After, cancellation, and backoff fixtures.
- U3: `siumai-transport` all-feature nextest/clippy/rustdoc; event-order and redaction contract tests.
- U4: `siumai-transport` all-feature nextest/clippy plus facade contract, doctest, clippy, and no-default `transport` feature checks; bounded local CONNECT, resolver/peer-policy, credential-phase, and unsupported-destination fixtures; no live network or credentials.
- U5: `siumai-mcp` all-feature nextest/clippy; bounded local CONNECT/SSE fixtures covering POST/GET/DELETE and never-replay semantics.
- U6: compatibility engines plus OpenAI, Anthropic, Gemini, and Vertex nextest/clippy; no-default/feature checks for HTTP, Responses WebSocket, and Realtime surfaces.
- U7: remaining provider package nextest/clippy in serial package groups; representative native/compatible observer and settings fixtures.
- U8: facade nextest, doctest, clippy, no-default and selected feature checks; cargo metadata inspection.
- U9: Direct-milestone documentation/doctest checks, Registry/setter deletion searches, formatting, and diff hygiene.
- U10: proxy/MCP documentation/doctest checks, proxy-surface searches and sentinel scan, formatting, and diff hygiene.

### Integration and Release Gates

Run Cargo commands serially and reuse the workspace target directory. Milestone A runs every applicable item except the MCP/proxy-specific U4-U5 checks and closes through U9. Full-plan completion adds U4-U5, their facade checks, and U10 before running the complete matrix below.

1. Run the workspace formatting check without rewriting unrelated shared-worktree changes.
2. Run nextest for core, transport, Registry, runtime, server, and—only for the full plan—MCP with all relevant features and one test thread.
3. Run nextest for both compatibility engines and the flagship OpenAI, Anthropic, Gemini, and Vertex providers, then run the remaining affected providers in serial package groups.
4. Run the facade nextest suite and facade rustdoc/doctests for the selected milestone.
5. Run clippy with warnings denied for the affected foundation, compatibility, provider, MCP, and facade packages in serial groups.
6. Check the facade with no default features, the standalone transport feature, representative individual provider/Registry/runtime features, OpenAI Responses WebSocket, OpenAI Realtime, and the all-provider combination.
7. Inspect Cargo metadata after feature/dependency changes to confirm the optional graph and absence of cycles.
8. Finish with working-tree and staged diff whitespace checks as applicable.

No live provider, billable, credentialed, destructive, public proxy, or external MCP test is part of the gate.

### Review Gates

- Security review: proxy/provider credential phase separation, DNS trust transfer, endpoint/replay preservation, environment isolation, redaction, and unsupported-surface rejection.
- API review: settings names, facade curation, timeout/retry semantics, migration completeness, and removal of speculative middleware/setters.
- Reliability review: deadline resolution exactly once, effective attempt budget, cancellation/backoff races, observer final outcome, composite-provider propagation, and MCP never-replay behavior.
- Architecture review: Registry remains lookup-only; no raw transport interceptor, universal provider trait, host policy, heuristic middleware, or duplicate settings owner appears.
- Agent parity review: provider HTTP and remote MCP share the explicit proxy route; runtime/Agent receive the same resolved deadline and provider settings without a second middleware pipeline.

---

## Definition of Done

### Milestone A — Direct infrastructure

- [ ] U1-U3 and U6-U9 satisfy their requirements, test scenarios, and verification outcomes in dependency order.
- [ ] All configured provider HTTP builders consume `ProviderHttpTransportSettings` through the final `with_http_transport_settings(...)` entry point.
- [ ] Composite providers share one `ProviderTransport` only across branches with identical endpoint, credential audience, auth/signing owner, HTTP route/settings, and network mechanism; deterministic fixtures prove isolation whenever any dimension differs.
- [ ] Duplicated provider-level HTTP settings setters are deleted; only genuinely session/lifecycle-specific controls remain.
- [ ] Relative timeout resolves once and caller attempt caps cannot increase retry or replay authority.
- [ ] Direct mode ignores environment proxies and preserves current endpoint/DNS/replay/credential behavior.
- [ ] Transport observer events are available across providers, payload-free, and distinguish stream establishment from protocol completion.
- [ ] `RegistryMiddleware`, its stack/builder/snapshot APIs, facade exports, and behavior tests are absent outside migration notes.
- [ ] Facade-only users can name every configuration type required by facade provider/runtime builder methods.
- [ ] No new heuristic, global telemetry registry, raw client/fetch hook, business routing/fallback/cache policy, universal provider trait, source parser, or capability matrix exists in the diff.
- [ ] Architecture, migration, README/example, changelog, and rustdoc text agree with the Direct public API; proxy support is clearly deferred rather than partially claimed.
- [ ] Milestone A Verification Contract gates pass serially; no live or credentialed test was run.
- [ ] Direct-migration transitional code, duplicate helpers, obsolete tests, and stale documentation are removed rather than left dormant.

### Full Plan — Direct plus explicit CONNECT

- [ ] R1-R36 are implemented and every acceptance example has a deterministic offline fixture or compile contract.
- [ ] U1-U10 satisfy their requirements, test scenarios, and verification outcomes in dependency order.
- [ ] Every Milestone A item remains true after the proxy extension.
- [ ] Trusted HTTPS CONNECT works for provider HTTP and streamable HTTP MCP with separate redacted Basic credentials and explicit DNS trust delegation.
- [ ] Invalid combinations on proxy-aware HTTP surfaces fail before network I/O; WebSocket, Realtime, and provider-returned external-download APIs do not accept or inherit the HTTP route and are documented as direct-only/deferred for proxy support.
- [ ] Facade-only users can construct the trusted route and proxy credential without importing execution/auth internals.
- [ ] Architecture, migration, README/example, changelog, and rustdoc text agree with the complete public API and trust model.
- [ ] All Verification Contract gates pass serially; no live, credentialed, public-proxy, or external-MCP test was run.
- [ ] Dead transitional code, abandoned proxy/middleware experiments, duplicate helpers, obsolete tests, and stale documentation discovered during implementation are removed rather than left dormant.
