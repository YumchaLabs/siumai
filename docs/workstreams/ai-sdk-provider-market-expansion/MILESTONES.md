# AI SDK Provider Market Expansion — Milestones

Status: Draft
Last updated: 2026-05-25

## M0 — Scope Freeze

Exit criteria:

- npm download evidence is captured with the exact date window.
- AI SDK package candidates are tied to local `repo-ref/ai/packages/*` sources.
- Siumai's current support state is documented without reopening closed workstreams.

Gate:

- Documentation consistency check across DESIGN.md, TODO.md, WORKSTREAM.json, HANDOFF.md, and EVIDENCE_AND_GATES.md.

## M1 — Gateway Contract Proof

Exit criteria:

- Gateway endpoints, headers, auth modes, model family surfaces, provider options, and metadata behavior are inventoried.
- The lane chooses one of:
  - implement a minimal Gateway provider proof,
  - split a dedicated Gateway implementation workstream,
  - or defer with concrete blockers.

Gate:

- Gateway inventory reviewed for architecture boundary fit.
- Gateway minimal proof landed with focused no-network tests for URL/header/body, language stream, embedding,
  registry, catalog, and public facade behavior. Media/admin/OIDC/tools/catalog expansion remains follow-on.

## M2 — Cerebras Support

Exit criteria:

- Cerebras is available through Siumai's provider surface or explicitly deferred with evidence.
- AI SDK quirks are handled or documented:
  - assistant reasoning history uses Cerebras `reasoning` rather than shared `reasoning_content`;
  - structured output plus `tool_calls` finish behavior is normalized if applicable.

Gate:

- Focused nextest for Cerebras preset/model/public-surface tests.
- `cargo fmt --check` for touched crates.

## M3 — Mistral And Enterprise Deepening

Exit criteria:

- Mistral's dedicated AI SDK package surface is audited against the current Siumai preset/facade.
- Azure, Bedrock, and Google Vertex have either no high-value polish gaps or focused fixes/docs for discovered gaps.

Gate:

- Focused tests/docs for any changed provider contracts.
- No widening of `prelude::unified` without explicit design justification.

## M4 — Audio And Media Decision

Exit criteria:

- Deepgram, ElevenLabs, Fal, Replicate, and similar provider packages are ranked by priority.
- Any selected provider is split into a dedicated implementation task or workstream.
- Deferred media/audio packages have explicit rationale.

Gate:

- Decision note is linked from HANDOFF.md and WORKSTREAM.json follow-on list if it creates new work.

## M5 — Closeout

Exit criteria:

- All task ledger items are done, split, or explicitly deferred.
- Evidence gates are refreshed.
- HANDOFF.md names the next task or states the lane is closed.
