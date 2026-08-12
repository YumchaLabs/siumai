# Siumai Next Delivery Status

This page is the internal delivery summary for the breaking `0.11.0-beta.9` development line. It
is intentionally shorter than the implementation plan and does not replace the architecture or
provider-support documents.

Status date: 2026-08-12

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
Batches, token counting, hosted-tool replay, and bounded Skills metadata/version operations, Gemini
language and product resources, Alibaba/DeepSeek Anthropic-compatible Messages, MiniMax portable
media adapters, Kimi and ARK native breadth, xAI Files/image/video/speech/transcription, Groq audio
and Remote MCP, and Deepgram/ElevenLabs portable audio families. The relevant delivery commits
include `5832445a`, `86a73026`, `53982be0`, `d71acbce`, `af6f0558`, `3f6ecc85`, `0fa5c402`, and
`4c4d32ba`.

For Anthropic Skills, `claimed slice complete` covers bounded create uploads and the implemented
metadata/version operations only. Skill version-content download is `intentionally deferred`; this
checkpoint does not make a `provider platform complete` claim.

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
`997ff09e`, `26a641d1`, `4387e9a9`, `566ecd27`, and `bba1abff`. Their earlier focused serial
evidence remains useful for protocol-level provenance, while the final workspace-wide release
evidence is recorded below.

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
`a4fcec9f`, `3d95dbdd`, `748c21b9`, `0f004f39`, and `e4daa4d7`. Release hardening then added the
diagnostic, OpenAI, Anthropic, package, version, CI, and support-claim fixes in `981d966f`,
`92b00d9e`, `ffad41bc`, `81f3347b`, `71550fec`, `efc418cd`, and `8635bea0`.

### Final deterministic release evidence

The serial release verification completed on 2026-08-12 against the current workspace:

- 1,161 workspace tests with all features, plus the 361-test fast suite and 380-test fixed flagship
  OpenAI/Anthropic suite;
- workspace Clippy across all targets and features with warnings denied;
- OpenAI provider documentation, workspace documentation, and workspace doctests;
- Rust 1.88 workspace/all-target/all-feature MSRV checking;
- facade default-free checks for bare, OpenAI, Anthropic, all-provider, and combined Responses
  WebSocket/Realtime ownership, plus each flagship example with its exact provider feature;
- 27 repository-script tests and the architecture-boundary check;
- Cargo metadata proving all 26 workspace packages report `0.11.0-beta.9`;
- 511 Cargo-selected package paths through the bounded package-list checker, followed by a complete
  `cargo package --workspace --locked --allow-dirty -j 1` dry run with no publication;
- workspace formatting, unstaged diff, and staged diff whitespace checks.

### Opt-in live diagnostic

The authorized `sub2api` diagnostic ran again on 2026-08-12 after all deterministic gates passed.
At the status snapshot generated at 2026-08-12T10:35:38Z, the endpoint was not globally green:
`gpt-5.6-sol` had a successful latest probe, while `gpt-5.6-luna` had a failed latest probe. The
canary therefore fixed `gpt-5.6-sol` explicitly; status did not choose a fallback or relax a wire
dialect. No credential, live endpoint value, response text, tool argument, raw provider payload, response ID,
or provider close reason was recorded:

- Branded Chat direct and Chat streaming completed with usage; the stream produced one usage
  snapshot and preserved usage on its terminal response.
- Branded Responses direct completed with usage and exposed known cache-read and cache-write
  dimensions. The relay still abbreviates Responses terminal items, so branded strict Responses
  streaming correctly failed with `ErrorKind::Protocol`. The generic compatible Responses stream,
  with an explicitly selected compatible dialect, completed with terminal usage. Its compatible
  Chat stream also completed with terminal usage.
- Two explicit repeated cache calls settled as retryable `ErrorKind::RateLimited` and
  `ErrorKind::Unavailable`. This run therefore makes no cache-hit attribution and does not treat
  relay capacity as a caching or parser defect.
- A compatible Responses tool turn produced exactly one canonical caller-owned tool call. The
  projected assistant history plus `ToolResult` continued successfully, with zero history
  omissions. This validates the unified tool/history boundary without weakening native replay or
  executable parity checks.
- The Responses WebSocket failed during setup as sanitized, retryable `ErrorKind::Unavailable`.
  It did not expose a provider-controlled close reason and is not evidence that multi-turn
  WebSocket continuation works on this relay.

The live run exposed one ergonomic gap rather than a semantic defect: a generic compatible caller
could not previously preserve an already validated RFC 6598 `EndpointConfig` through a public
constructor. `OpenAiCompatibleProfile::custom_endpoint` now accepts that transport policy while
retaining generic claims, a custom replay audience, and a strict Responses default. Callers must
explicitly select a compatible dialect only when their relay fixtures prove it.

The temporary canary lived outside the repository, used fixed time and event bounds, and was
removed after the run. Live provider canaries remain opt-in diagnostics. They are not release gates
and must not turn relay capacity, quota, or upstream availability into parser or API claims.
