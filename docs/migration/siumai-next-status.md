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

## Checkpoint C — flagship product surfaces and release readiness

Status: complete.

The current line includes the planned typed OpenAI lifecycle/media slices, Gemini language and
product resources, Alibaba/DeepSeek Anthropic-compatible Messages, MiniMax portable media adapters,
Kimi and ARK native breadth, xAI Files/image/video/speech/transcription, Groq audio and Remote MCP,
and Deepgram/ElevenLabs portable audio families. The relevant delivery commits are `5832445a`,
`86a73026`, `53982be0`, `d71acbce`, and `af6f0558`.

Release hygiene is complete. The root and affected crate changelogs, breaking migration guide,
delivery status, and Hajimi handoff now describe one coherent API line. The final serial gates passed:

- 986 workspace/all-features nextest cases;
- workspace/all-targets/all-features Clippy with warnings denied;
- workspace doctests and rustdoc generation;
- Rust 1.88 workspace/all-targets/all-features check;
- no-default `all-providers` facade Clippy and the optional runtime JSON Schema feature check;
- architecture policy, metadata, package-content lists for 24 changed packages, formatting, relative
  documentation links, and final diff checks.

Live provider canaries remain opt-in diagnostics. They are not release gates and must not be used to
turn relay capacity, quota, or upstream availability into parser or API claims.
