# Fearless Module Deepening — Milestones

Status: Complete
Last updated: 2026-05-19

## M0 — Scope And Evidence Freeze

Exit criteria:

- The architecture review findings are captured as concrete Module deepening candidates.
- The first executable slice is selected.
- Relevant ADRs, architecture docs, and completed workstreams are linked.
- Closeout policy and gate expectations are explicit.

Primary evidence:

- `docs/workstreams/fearless-module-deepening/DESIGN.md`
- `docs/workstreams/fearless-module-deepening/TODO.md`
- `docs/workstreams/fearless-module-deepening/EVIDENCE_AND_GATES.md`

## M1 — Spec Provider Residue Removed

Exit criteria:

- Provider-defined tool catalog helpers no longer have their canonical implementation in
  `siumai-spec`.
- Provider/protocol-owned catalog Modules expose the canonical helpers.
- Facade and provider extension import paths remain stable or have explicit migration notes.
- Closed provider classification is replaced, deprecated, or split behind a provider-id-first seam.

Primary gates:

- `cargo nextest run -p siumai-spec --no-default-features`
- Protocol crate targeted gates for OpenAI, Anthropic, and Gemini hosted tools.
- Facade public-surface import guards.

## M2 — Compatibility Interfaces Isolated

Exit criteria:

- `LlmClient` and generic-client downcast Interfaces are physically scoped as compatibility.
- Stable registry family handles continue to avoid compat/downcast paths for primary execution.
- Old broad registry construction helpers are removed, deprecated, or quarantined behind compat
  names.

Primary gates:

- `cargo nextest run -p siumai-registry --no-default-features`
- `cargo nextest run -p siumai-registry --features builtins --no-default-features`
- Facade architecture boundary tests.

## M3 — Protocol And Bridge Modules Deepened

Exit criteria:

- Request normalization has narrower protocol-pair Adapter seams for behavior that genuinely varies.
- OpenAI protocol internals are split around behavior, not file size.
- Vercel fixture parity is unchanged.

Primary gates:

- `cargo nextest run -p siumai-bridge --features openai,anthropic,google --no-fail-fast`
- `cargo nextest run -p siumai-protocol-openai --all-features --no-fail-fast`
- Relevant facade bridge and OpenAI fixture tests.

## M4 — Legacy Content Direction Chosen

Exit criteria:

- The lane either lands a small directional `ContentPart` adapter proof slice or opens a narrower
  follow-on for the breaking public-shape work.
- Lossless/lossy conversion policy is tested, not implicit.
- ADR-0008 is either satisfied for the chosen slice or referenced by the follow-on.

Primary gates:

- `cargo nextest run -p siumai-spec --no-default-features prompt`
- `cargo nextest run -p siumai-bridge --features openai,anthropic,google request`

## M5 — Closeout

Exit criteria:

- Every TODO task is complete, intentionally deferred, or split into a child workstream.
- `EVIDENCE_AND_GATES.md` has fresh verification evidence.
- Architecture docs reflect the shipped ownership rules.
- `WORKSTREAM.json` status is updated.
- `HANDOFF.md` records residual risks and the next recommended action.

Closeout result:

- FMD-010 through FMD-110 are complete.
- Fresh closeout verification on 2026-05-19 passed for formatting, spec, core boundary, bridge,
  registry all-provider, OpenAI/Anthropic/Gemini protocol crates, facade public surface, and
  all-provider provider public-path parity.
- The only intentionally deferred item is the breaking public `ContentPart` namespace move, which
  should be opened as a separate compatibility-break workstream rather than kept in this lane.
