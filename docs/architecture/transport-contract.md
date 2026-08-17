# Transport Contract

- Status: Current repository contract
- Date: 2026-08-17
- Owner: `siumai-transport`

## Purpose

Provider transports are a security and lifecycle boundary, not a collection of
HTTP helpers. Siumai has one Direct execution path for authenticated provider calls and a separate
structurally unauthenticated path for provider-returned resources.

Protocol crates still own request/response codecs, provider error envelopes, and
stateful stream decoders. Provider crates own credentials, signing, operation
replay declarations, stable request validation, and typed options. Mutable model
lifecycle or capability advice is not transport or request-execution authority.

## Direct HTTP Settings

`siumai-transport` owns the cloneable `ProviderHttpTransportSettings` value. Facade users reach the
same type as `siumai::transport::ProviderHttpTransportSettings`; the facade's `transport` feature
owns that curated namespace, and every facade provider feature activates it.

The value configures provider-owned stateless HTTP limits, provider retry policy, connect/call/read
timeouts, and one synchronous payload-free attempt observer. Every configured provider and both
compatibility engines accept it through `with_http_transport_settings(...)`.

The following authorities deliberately remain separate inputs and cannot be overridden through
provider options or request bodies:

- endpoint policy and the exact credential audience;
- credential and signing application;
- DNS resolution and provider retry classification;
- per-operation replay proof;
- the relative request target, headers, and rebuildable body.

A composite provider may clone one `ProviderTransport` only across branches with the same exact
endpoint, credential audience and auth/signing owner, settings, and network mechanism. Different
technical identities use different transports even when they share one provider name.

Milestone A exposes Direct networking only. A custom endpoint remains a caller-selected reverse
gateway destination with its own endpoint and replay audience; it is not a forward proxy. Explicit
forward proxy/CONNECT routing and the corresponding MCP HTTP route adapter are deferred to
Milestone B. No partial proxy type or proxy support claim is part of the Direct contract.

## Authenticated API Calls

`ProviderTransport` is built once per exact configured HTTP runtime identity and cloned into
lightweight model handles. Construction is synchronous and creates one shared client, connection
pool, settings snapshot, queue, and in-flight limit.

Every call supplies an immutable `RequestPlan`:

- a relative `RequestTarget` that cannot change origin or escape the base path;
- non-credential headers that cannot override transport-owned fields;
- a closed rebuildable body (`Empty`, bounded bytes/JSON, or deterministic multipart);
- one closed `ReplaySafety` proof.

Authentication is applied inside the transport after the final target and body are
known. An `AuthApplier` may produce sensitive headers or bounded secret query
parameters for signing/API-key protocols. It cannot change host, length, transfer
encoding, proxy authentication, or the credential audience.

The client explicitly disables:

- automatic redirects;
- environment/system proxies;
- automatic referer forwarding;
- reqwest's protocol-level retry policy.

This keeps the library's attempt budget authoritative. Direct mode does not discover or honor
environment/system proxy variables. A future trusted-proxy mode must be modeled as an explicit
transfer of destination DNS/peer enforcement to that proxy; it must not weaken the Direct client.

## Replay

The default proof is `Never`. A request that may have reached the service is not
replayed merely because no response or stream event was observed.

`SemanticallyIdempotent` is used only when repeating the remote operation has the
same effect. `IdempotencyKey` is used only for a provider-documented header and a
rebuildable body. The transport creates a fresh key when `execute` starts, reuses
it byte-for-byte across that call's attempts, and never stores it in `RequestPlan`.

Network failures, one supported 401 credential refresh, 429 responses, and selected transient 5xx
responses consume the same total-attempt budget. `CallOptions::with_max_attempts` supplies a caller
ceiling for each logical provider HTTP call; `without_retry()` is the one-attempt convenience. The
effective budget is the minimum of provider policy and caller cap only after replay proof is
checked. `ReplaySafety::Never` therefore always permits one attempt, and a caller cap never creates
an idempotency key or retryable status. Once a byte stream is returned, the transport never replays
or reconnects it. Protocol-specific resume is a separate experimental operation.

`RetryPolicy::max_server_delay` is independent of local exponential `max_backoff`. Standard
`Retry-After` advice above that ceiling, or advice that would reach or exceed the remaining call
deadline, declines the retry and returns the current typed failure. Transport never retries earlier
than the server requested.

## Call Timing

`CallOptions::with_timeout` records a relative timeout. It starts when the outer public family call
or runtime run accepts the options, not when the options value is constructed. That boundary calls
`CallOptions::resolve_deadline()` once, clears the relative form, and propagates the resulting
absolute deadline unchanged through validation, queueing, authentication refresh, retry backoff,
stream establishment, runtime steps, and structured-output repair. When an absolute deadline is
also present, the earlier deadline wins. Zero or platform-unrepresentable durations fail with a
typed input error before network submission.

Every Siumai-owned family implementation and wrapper performs this idempotent resolution. Because
the family traits are open, a third-party implementation must invoke the public helper at its own
outer call boundary; Siumai does not claim to enforce that behavior in external implementations.

## Endpoints And DNS

Credentials are bound to an exact scheme, normalized host, and effective port.
Trailing-dot hosts, user information, fragments, absolute request targets,
backslashes, and path traversal segments are rejected.

`Official` requires an `OfficialOrigin` created from provider-owned static metadata,
and the configured endpoint must match its exact scheme, normalized host, and
effective port. Both `Official` and `PublicCustom` require HTTPS and globally
routable DNS answers. `LocalExplicit` permits HTTP/HTTPS only for one exact
caller-selected grant: loopback, RFC 1918 or IPv6 unique-local private space,
link-local space, or RFC 6598 shared address space (`100.64.0.0/10`). The RFC 6598
grant does not widen any adjacent scope and treats IPv4-mapped IPv6 as the same
address. Cleartext local transport is an explicit caller trust decision; HTTPS/WSS
should be preferred whenever the deployment supports it. Every DNS answer is
filtered in reqwest's actual connector resolver, so a preflight/connection
rebinding gap cannot bypass policy. Literal IPs are checked at construction, and a
reported connected peer is checked again after the handshake.

WebSocket endpoints use the same address policy while preserving `ws`/`wss` as a
distinct credential audience. `WebSocketTransport` owns resolution, direct TCP
connection to a validated address, peer verification, TLS/SNI, authentication,
handshake, message limits, admission permits, I/O deadlines, and drop cancellation.
It does not expose separate resolve/connect validation steps that callers could
forget or reorder.

## Attempt Observation

The observer is read-only, synchronous, bounded, and payload-free. One opaque `TransportCallId`
correlates the complete attempt loop without carrying provider, model, route, account, tenant, URL,
query, header, credential, prompt, tool, request-body, response-body, or provider-error data. A host
that needs attribution wraps the observer per configured provider and attaches its own context
outside `TransportEvent`; transport does not invent that identity.

The lifecycle is exact:

1. `AttemptBudgetResolved` reports configured and effective attempt budgets and the limiting
   authority.
2. Each submission emits `AttemptStarted`, followed by `ResponseHeadReceived` when applicable.
3. A retry emits either `RetryScheduled` or `RetryDeclined` with structural reason and delay.
4. Exactly one `AttemptLoopFinished` reports response return, stream establishment, failure,
   cancellation, or timeout.

For buffered calls, observation ends when the bounded response is ready to return. For streaming
calls, it ends when the byte stream is established. Later body consumption, protocol decoding, and
semantic stream termination are outside the HTTP attempt observer.

## Resources

`ResourceDownloader` cannot receive an `AuthApplier` or an external `reqwest::Client`.
It follows redirects manually within a fixed budget, validates each new URL and DNS
answer, disables proxy/referer/retry behavior, and never copies provider credentials.
Resources authorized through any `LocalExplicit` grant keep redirects on the exact
original scheme, normalized host, and effective port. Same-origin redirects remain
available, while a loopback, private, link-local, or RFC 6598 response cannot pivot
to another local service or downgrade its scheme. Public resource redirects remain
subject to validation on every hop.
Network and data URLs share the decompressed response-byte limit and the same
admission controls. Data URLs are decoded in bounded blocking work that observes
cancellation and deadlines while retaining its permits. Declared media type and
independently sniffed media type are returned as evidence rather than silently
trusted.

Provider response URLs must enter this downloader by default. A provider resource
that genuinely requires authentication must instead be modeled as an exact-origin
typed API operation; suffix matching and query-key forwarding are forbidden.

`ProviderHttpTransportSettings` never propagates implicitly into this downloader, provider
WebSocket/Realtime sessions, media sessions, jobs, or MCP. OpenAI Realtime and Responses WebSocket
retain their dedicated limits and connect/session/I/O/turn timeouts. Alibaba video materialization
retains dedicated direct-download limits and connect/download/read timeouts. These controls remain
independent because their lifecycle and trust boundary differ from stateless provider HTTP.

## Limits And Cancellation

`TransportLimits` bounds request bodies, decompressed responses, headers, frames,
events, multipart parts, redirects, queued calls, connections, and in-flight work.
Queues use bounded admission and cancellation-aware semaphore waits. Buffered calls
hold their permits until the response body is fully consumed. Established streams
own their response, cancellation child, and permits; drop cancels the child and
releases all resources without creating a detached reader task.

Provider HTTP calls have a finite default total deadline and a per-read idle
timeout; explicit earlier call deadlines still win. WebSocket sessions likewise
have finite session and per-I/O defaults. Builders may tune these values but reject
zero durations. Limit validation also rejects capacities above hard safety ceilings
before semaphore or buffer construction.

Default diagnostics include only fixed public messages and allowlisted bounded headers. Credential
headers use `HeaderValue::set_sensitive(true)`. Provider credentials, endpoint queries, response
bodies, prompts, and tool input/output never appear in ordinary `Debug`, `Display`, serialized
errors, or observer events. Raw bodies, headers, signed URLs, and underlying errors require explicit
sensitive access and are captured only within fixed limits.

MCP applies the same ownership rules at its integration boundary. Streamable HTTP
disables automatic redirects, proxy discovery, client retries, and rmcp session
reinitialization. JSON responses and error bodies are byte-bounded before decoding;
SSE and stdio messages are incrementally bounded before protocol materialization.
Backend errors cross the public API through static operation phases, with their
original source available only through the explicit sensitive accessor. Progress
notifications use a bounded broadcast queue; queue lag is observable, but a
drained long-lived session is never poisoned by a lifetime notification counter.
The current runtime executor boundary does not pass a cancellation token into an
MCP call, and rmcp's high-level tool helper does not expose its request handle;
therefore cancellation after dispatch is recorded as indeterminate rather than
being treated as proof that replay is safe.

Milestone A does not add an MCP proxy route. Streamable HTTP MCP remains Direct until the separately
reviewed Milestone B route adapter lands.

## Framing Boundary

SSE, JSONL, and WebSocket utilities enforce UTF-8/message boundaries, frame size,
aggregate event size, and event count. They do not recognize provider terminal
markers, translate tool events, repair JSON, or select a decoder by provider ID.
OpenAI, Anthropic, Gemini, and other protocols each retain one state machine and one
finish path above this layer.

## Migration State

This crate is the target transport contract for provider implementations. Provider packages use the
shared HTTP settings and transports or a narrowly documented provider-owned transport when a
protocol cannot fit the shared contract. New code must not reintroduce duplicated provider HTTP
setters, environment proxy discovery, raw client/fetch hooks, outer retries, or detached stream
readers that bypass endpoint, credential-audience, cancellation, replay, observation, or
resource-bound rules.
