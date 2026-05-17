# Stream Delta Lossless Boundary

Status: Closed
Last updated: 2026-05-17

## Why This Lane Exists

Issue #19 exposed a stream architecture invariant that was only implicit: model-generated text
deltas are payload, not framing. A newline-only or space-only delta is valid generated content and
must survive the raw transport, SSE parsing, protocol adapter, and field extraction path unchanged.

The immediate bug is fixed, but the architecture still needs a sharper seam so future provider
work does not reintroduce `trim`-based filtering in a shared helper.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
- Existing docs:
  - `docs/alignment/streaming-bridge-alignment.md`
  - `docs/workstreams/stream-metadata-parity-hardening/design.md`
  - `docs/workstreams/ai-sdk-structural-alignment/design.md`
- Reference implementation:
  - `repo-ref/ai/packages/provider-utils/src/parse-json-event-stream.ts`
  - `repo-ref/ai/packages/provider-utils/src/response-handler.ts`
  - `repo-ref/ai/packages/openai/src/chat/openai-chat-language-model.ts`
  - `repo-ref/ai/packages/openai-compatible/src/chat/openai-compatible-chat-language-model.ts`
- Related fix:
  - `c64b8980 fix(openai-compatible): preserve whitespace stream deltas`

## Problem

Before the fix, stream code made text-semantics decisions in multiple places:

- `siumai-core::StreamFactory` treated whitespace-only SSE `event.data` as empty framing.
- OpenAI-compatible content and reasoning extractors treated whitespace-only model fields as absent.
- Field extraction reused the same string helper shape for generated payload fields and control
  fields, even though those fields have different invariants.

That creates a shallow interface: callers need to remember which helper preserves generated text,
which helper normalizes control strings, and which layer is allowed to skip data. The risk is not
limited to OpenAI-compatible providers; the same mistake can appear in any provider that streams
text, reasoning, or tool argument deltas.

## Target State

When this lane closes:

- raw transport and SSE parsing only handle framing, stream-end markers, and parse errors;
- provider adapters own provider-specific wire semantics;
- generated text and reasoning delta extraction is lossless by construction;
- control field extraction remains free to trim or normalize where appropriate;
- tests prove whitespace-only and empty generated deltas cannot be dropped by shared streaming
  infrastructure;
- docs state the invariant so future provider work has one place to follow.

Closed state: this target is implemented for the OpenAI-compatible chat and completion streaming
seams touched by this lane. Completion streaming was included because the all-features provider gate
exposed the same stream-end and field-presence boundary problem.

## In Scope

- OpenAI-compatible field accessor and stream converter semantics.
- `siumai-core` stream factory framing behavior around empty raw SSE data.
- Focused OpenAI/OpenAI-compatible/core streaming tests for whitespace-only and empty deltas.
- Workstream and architecture notes documenting the invariant.

## Out Of Scope

- Rewriting all provider stream implementations in one pass.
- Changing public `ChatStreamEvent` or `ChatStreamPart` variants.
- Broad provider metadata parity work.
- Introducing new crates only for this invariant.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Generated text/reasoning deltas should be lossless, including whitespace-only values. | High | Vercel OpenAI adapter emits `delta.content` when it is `!= null`; issue #19 reproduced newline loss. | If wrong, clients expecting normalized text would need an opt-in normalization layer above streaming. |
| Empty raw SSE data is framing, not generated content. | High | SSE keepalive/blank data frames are outside parsed JSON fields. | If wrong, raw frame handling must expose a lower-level byte/event stream API. |
| Empty generated string fields may be semantically useful for parity, but whitespace-only fields are already user-visible. | Medium | Vercel OpenAI tests include an empty `text-delta`; OpenAI-compatible reference skips empty strings but preserves whitespace. | We may choose field-presence semantics for closer OpenAI parity, or defer exact empty-string parity if it changes local stream consumers. |
| The first valuable slice can stay inside OpenAI-compatible plus shared core tests. | High | Issue #19 and current helper shape both live there. | If other providers reveal the same bug, split a follow-on audit task rather than widening this lane silently. |

## Architecture Direction

Adopt the same seam shape as Vercel AI SDK, adapted to Rust:

- shared stream utilities parse bytes/SSE/JSON and skip only true framing;
- adapters map provider wire chunks into Siumai stream events;
- generated delta extraction uses field-presence or exact-string semantics and never applies
  `trim`;
- control extraction uses explicit control-oriented helpers where normalization is intended.

The deepened module here is not a new abstraction for every provider. The immediate leverage comes
from naming the two string semantics and making the existing OpenAI-compatible accessor impossible
to use ambiguously.

## Closeout Condition

This lane can close when:

- the target state is implemented for the OpenAI-compatible seam touched by issue #19;
- focused core/protocol/provider gates pass;
- docs and evidence record the invariant;
- any remaining provider-wide audit is either completed, deferred with rationale, or split into a
  follow-on workstream.

Closeout: no follow-on was split. The provider-wide audit can stay opportunistic; the known
OpenAI-compatible chat/completion regression class is covered by focused tests and all-features
provider validation.
