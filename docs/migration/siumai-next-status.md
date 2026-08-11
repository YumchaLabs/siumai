# Siumai Next Delivery Status

This page is the internal delivery summary for the breaking `0.11.0-beta.9` development line. It
is intentionally shorter than the implementation plan and does not replace the architecture or
provider-support documents.

Status date: 2026-08-11

The completion label `claimed slice complete` means the declared beta-release slice is implemented
and covered by its recorded deterministic gates. It is not a `provider platform complete` claim.
Known product surfaces outside the declared slice are `intentionally deferred`.

## Checkpoint A — semantic trust boundary

Status: claimed slice complete.

The shared language contract now has one canonical success-or-error stream settlement. Caller-owned
tool calls use checked, bounded JSON input; provider-executed and unknown native items remain opaque
provider data. Typed provider failures preserve category, retryability, delay, and a sanitized public
message. Replay-critical identity and history projection are validated at the semantic boundary.

The checkpoint was delivered through the core and protocol refactors leading to `b7502013`,
`8b59c658`, and the subsequent OpenAI/Anthropic correctness commits. It is consumed by runtime,
server, MCP, compatibility engines, and the facade rather than by a parallel consumer-only adapter.

## Checkpoint B — provider ownership and API modes

Status: claimed slice complete.

Branded Moonshot AI and Volcengine ARK surfaces now have provider-owned crates instead of public
profiles inside the generic OpenAI-compatible engine. Alibaba endpoint provenance, Gemini's
product-level provider identity, and provider/replay ownership are explicit. Caller-selected
endpoints do not inherit verified provider claims merely because a transport policy label resembles
an official endpoint. Technical workspace, project, and location inputs remain caller-controlled
addressing data; Siumai does not maintain a region or availability catalog.

This checkpoint includes the provider extraction and Gemini work (`12027215`, `bc59b59f`,
`1c838d09`, `796b0065`, and `46013147`).

## Checkpoint C — flagship product surfaces

Status: claimed slice complete.

The current line includes the planned typed OpenAI lifecycle/media slices, Anthropic Files, Message
Batches, token counting, hosted-tool replay, and Skills lifecycle, Gemini language and product
resources, Alibaba/DeepSeek Anthropic-compatible Messages, MiniMax portable media adapters, Kimi and
ARK native breadth, xAI Files/image/video/speech/transcription, Groq audio and Remote MCP, and
Deepgram/ElevenLabs portable audio families. The relevant delivery commits include `5832445a`,
`86a73026`, `53982be0`, `d71acbce`, `af6f0558`, `3f6ecc85`, `0fa5c402`, and `4c4d32ba`.

The earlier revival workspace gate established a broad provider baseline. It remains historical
evidence for that checkpoint, not proof that every future provider field, event shape, or product
surface is permanently complete.

## Checkpoint D — OpenAI conversation conformance

Status: claimed slice complete.

Chat tool identity continuation, trailing usage, and bounded metadata now match observed protocol
ordering. Responses SSE reconstructs only policy-permitted abbreviated terminal fields and exposes
the canonical terminal resource separately from the exact native event. Prompt-cache intent is one
node-scoped wire breakpoint; provider-side cache read/write selection is not predicted locally, and
TTL and deprecated retention remain distinct typed wire controls. RFC 6598 relay endpoints require
an exact explicit transport grant.

The OpenAI provider now also owns an experimental persistent Responses WebSocket session behind the
independent `openai-responses-websocket` feature. HTTP and WebSocket turns share request
normalization and one Responses semantic decoder. Sessions enforce one active response, typed
turn/session cancellation, bounded queues and deadlines, response identity, sequential continuation,
native-only warm-up, and conservative close behavior. Only the provider-owned official endpoint
publishes the dated native support claim.

The implementation units are `fe1295c9`, `d2409ee3`, `eb8c27d1`, `73439125`, `e2dce58d`,
`997ff09e`, `26a641d1`, `4387e9a9`, `566ecd27`, and `bba1abff`. The focused serial release evidence
recorded on 2026-08-10 for this checkpoint is:

- 112 OpenAI protocol tests;
- 105 OpenAI provider tests;
- 80 transport tests;
- 25 facade tests, including the dedicated no-default WebSocket feature contract;
- Clippy with warnings denied for the OpenAI protocol, OpenAI provider, transport, and facade;
- facade doctests, no-default feature compilation, formatting, architecture policy, and diff checks.

## Checkpoint E — validation ownership and forward compatibility

Status: claimed slice complete.

Mutable model catalogs and lifecycle hints no longer control Registry construction or provider
execution. Explicit typed provider intent reaches the final wire or fails with a typed structural
error; model-name allowlists no longer silently remove options. Provider call options are bounded,
ordered patches bound to the exact configured instance, family, API mode, and optional route.
Runtime keeps route/model/step/call precedence private, and provider modes without a reviewed raw
body policy reject raw options.

Portable language success now has one completed-or-incomplete termination axis. Direct failures use
`LanguageCallError`; established failed or cancelled streams preserve only bounded non-executable
partial output; usage events declare snapshot or delta semantics and runtime settles each provider
call once. Runtime snapshots use schema version 6.

The implementation units are `817cdedf`, `119562a5`, `4ca3a764`, `434e11d0`, `29f2b1aa`,
`a4fcec9f`, `3d95dbdd`, `748c21b9`, `0f004f39`, and `e4daa4d7`. The serial verification baseline
recorded on 2026-08-10 includes:

- 1,058 workspace tests with all features;
- workspace Clippy across all targets and features with warnings denied;
- workspace formatting and diff checks;
- 19 focused OpenAI-compatible tests and its all-target/all-feature Clippy lane after adding the
  public validated custom-endpoint constructor.

### Opt-in live diagnostic

The authorized `sub2api` diagnostic ran on 2026-08-10. Its status endpoint was partially green, not
globally green: the selected `gpt-5.6-sol` lane was healthy while other listed models still had
recent failures. No credential, response text, tool argument, raw provider payload, or response ID
was recorded:

- Branded OpenAI Responses direct and Chat direct completed with usage. Chat streaming retained the
  late usage-only chunk. The repeated cache probes preserved cache-read/write telemetry, but the
  first call already reported a cache read, so this run cannot attribute the observation to the
  second call or to one cache key.
- The relay rejected `previous_response_id` continuation as an OpenAI request error. That is a relay
  product gap, not evidence for a model-policy gate or a portable contract change.
- The relay abbreviates Responses terminal items. The branded `OpenAiProvider` correctly rejected
  that stream under the strict OpenAI wire baseline. The generic OpenAI-compatible provider, with
  an explicitly selected compatible dialect, completed both Responses and Chat streams and retained
  usage.
- A compatible Responses tool turn produced exactly one canonical caller-owned tool call. The
  projected assistant history plus `ToolResult` continued successfully, with zero history
  omissions. This validates the unified tool/history boundary without weakening native replay or
  executable parity checks.
- The Responses WebSocket handshake began a turn and then settled as sanitized, retryable
  `ErrorKind::Unavailable`. It did not expose the provider-controlled close reason and is not
  evidence that multi-turn WebSocket continuation works on this relay.

The live run exposed one ergonomic gap rather than a semantic defect: a generic compatible caller
could not previously preserve an already validated RFC 6598 `EndpointConfig` through a public
constructor. `OpenAiCompatibleProfile::custom_endpoint` now accepts that transport policy while
retaining generic claims, a custom replay audience, and a strict Responses default. Callers must
explicitly select a compatible dialect only when their relay fixtures prove it.

Live provider canaries remain opt-in diagnostics. They are not release gates and must not turn
relay capacity, quota, or upstream availability into parser or API claims.
