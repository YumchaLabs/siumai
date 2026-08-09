# Siumai Next Delivery Status

This page is the internal delivery summary for the breaking `0.11.0-beta.9` development line. It
is intentionally shorter than the implementation plan and does not replace the architecture or
provider-support documents.

Status date: 2026-08-09

## Checkpoint A — semantic trust boundary

Status: complete.

The shared language contract now has one canonical success-or-error stream settlement. Caller-owned
tool calls use checked, bounded JSON input; provider-executed and unknown native items remain opaque
provider data. Typed provider failures preserve category, retryability, delay, and a sanitized public
message. Replay-critical identity and history projection are validated at the semantic boundary.

The checkpoint was delivered through the core and protocol refactors leading to `b7502013`,
`8b59c658`, and the subsequent OpenAI/Anthropic correctness commits. It is consumed by runtime,
server, MCP, compatibility engines, and the facade rather than by a parallel consumer-only adapter.

## Checkpoint B — provider ownership and API modes

Status: complete.

Branded Moonshot AI and Volcengine ARK surfaces now have provider-owned crates instead of public
profiles inside the generic OpenAI-compatible engine. Alibaba endpoint provenance, Gemini's
product-level provider identity, and provider/replay ownership are explicit. Caller-selected
endpoints do not inherit verified provider claims merely because a transport policy label resembles
an official endpoint. Technical workspace, project, and location inputs remain caller-controlled
addressing data; Siumai does not maintain a region or availability catalog.

This checkpoint includes the provider extraction and Gemini work (`12027215`, `bc59b59f`,
`1c838d09`, `796b0065`, and `46013147`).

## Checkpoint C — flagship product surfaces

Status: complete.

The current line includes the planned typed OpenAI lifecycle/media slices, Gemini language and
product resources, Alibaba/DeepSeek Anthropic-compatible Messages, MiniMax portable media adapters,
Kimi and ARK native breadth, xAI Files/image/video/speech/transcription, Groq audio and Remote MCP,
and Deepgram/ElevenLabs portable audio families. The relevant delivery commits are `5832445a`,
`86a73026`, `53982be0`, `d71acbce`, and `af6f0558`.

The earlier revival workspace gate established a broad provider baseline. It remains historical
evidence for that checkpoint, not proof that every future provider field, event shape, or product
surface is permanently complete.

## Checkpoint D — OpenAI conversation conformance

Status: complete.

Chat tool identity continuation, trailing usage, and bounded metadata now match observed protocol
ordering. Responses SSE reconstructs only policy-permitted abbreviated terminal fields and exposes
the canonical terminal resource separately from the exact native event. Prompt-cache intent is
node-scoped, current-write budgets are mode-specific, and TTL and retention remain independent typed
controls. RFC 6598 relay endpoints require an exact explicit transport grant.

The OpenAI provider now also owns an experimental persistent Responses WebSocket session behind the
independent `openai-responses-websocket` feature. HTTP and WebSocket turns share request
normalization and one Responses semantic decoder. Sessions enforce one active response, typed
turn/session cancellation, bounded queues and deadlines, response identity, sequential continuation,
native-only warm-up, and conservative close behavior. Only the provider-owned official endpoint
publishes the dated native support claim.

The implementation units are `fe1295c9`, `d2409ee3`, `eb8c27d1`, `73439125`, `e2dce58d`,
`997ff09e`, `26a641d1`, `4387e9a9`, and `566ecd27`. The focused serial release evidence for this
checkpoint is:

- 110 OpenAI protocol tests;
- 103 OpenAI provider tests;
- 80 transport tests;
- 25 facade tests, including the dedicated no-default WebSocket feature contract;
- Clippy with warnings denied for the OpenAI protocol, OpenAI provider, transport, and facade;
- facade doctests, no-default feature compilation, formatting, architecture policy, and diff checks.

### Opt-in live diagnostic

The authorized `sub2api` diagnostic ran on 2026-08-09 after its status endpoint reported green. No
credential, endpoint value, response text, tool argument, provider ID, or raw payload was recorded:

- Chat streaming produced one canonical local tool call, and direct continuation accepted the
  projected assistant/tool history. The upstream direct response itself carried null content;
  Siumai preserved that empty result instead of fabricating text.
- Responses text and tool SSE completed with canonical terminal responses after compatible recovery
  restored metadata omitted by the abbreviated terminal event. Present semantic conflicts remain
  deterministic protocol errors.
- Repeated Responses calls using typed `prompt_cache_key` plus 24-hour retention produced a cache
  hit on the second call. The relay accepted typed TTL options but returned HTTP 502 whenever a
  content-level explicit breakpoint was present, so no named claim is made for that custom relay's
  explicit-breakpoint fidelity.
- The Responses WebSocket handshake returned HTTP 101 and accepted `response.create`, then the relay
  closed with standard code 1013. Siumai now reports that as sanitized, retryable
  `ErrorKind::Unavailable` rather than `UnexpectedEof`. The operational close is not evidence that
  two-turn live continuation succeeded on this relay.

Live provider canaries remain opt-in diagnostics. They are not release gates and must not be used to
turn relay capacity, quota, or upstream availability into parser or API claims.
