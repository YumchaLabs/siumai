# Provider Native Extension Overrides — TODO

Status: Closed
Last updated: 2026-05-25

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[-]` intentionally deferred or split

## M0 — Scope And Inventory

- [x] PNEO-010 [owner=planner] [deps=none] [scope=docs/workstreams/provider-native-extension-overrides]
  Goal: Open the follow-on lane for provider-native file, skills, and music extension factory
  overrides.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree; `docs/workstreams/INDEX.md` includes the
  lane.
  Review: planner self-review.
  Evidence: workstream docs.
  Handoff: DONE. Workstream docs opened, indexed, and validated. First executable code task was
  PNEO-020.

- [x] PNEO-020 [owner=codex] [deps=PNEO-010] [scope=siumai-registry/src,siumai-registry/tests]
  Goal: Add a focused source guard that names each selected provider-native extension hook and
  rejects generic-client adapter fallback.
  Validation:
  `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_native_extension_hooks_bypass_generic_client_adapters --no-default-features --features openai,azure,anthropic,google,xai,minimaxi --no-fail-fast`
  Review: registry construction-boundary review.
  Evidence: failing-then-passing boundary test.
  Handoff: DONE. Added `provider_native_extension_hooks_bypass_generic_client_adapters`, first
  observed it fail on the missing Azure file hook, then turned it green after PNEO-030.

## M1 — Native Extension Hook Overrides

- [x] PNEO-030 [owner=codex] [deps=PNEO-020] [scope=siumai-registry/src/registry/factories]
  Goal: Override provider-native file, skills, and music extension hooks where the typed provider
  client already implements the target extension trait.
  Validation:
  `cargo check -p siumai-registry --tests --no-default-features --features openai,azure,anthropic,google,xai,minimaxi`
  Review: provider factory review.
  Evidence: typed-client projection in factory methods.
  Handoff: DONE. Azure OpenAI, OpenAI, Anthropic, Gemini, xAI, and MiniMaxi now return native typed
  clients from selected extension hooks.

## M2 — Evidence And Closeout

- [x] PNEO-040 [owner=codex] [deps=PNEO-020,PNEO-030] [scope=docs/workstreams/provider-native-extension-overrides]
  Goal: Record validation evidence, classify remaining defaults, and decide whether the lane can
  close or should split a follow-on.
  Validation:
  - `git diff --check -- docs/workstreams/provider-native-extension-overrides docs/workstreams/INDEX.md siumai-registry/src/registry/factories siumai-registry/tests/factory_architecture_boundary_test.rs`
  - final focused cargo gates from `EVIDENCE_AND_GATES.md`
  Review: final self-review.
  Evidence: updated workstream docs.
  Handoff: DONE. Lane closed. Speech/transcription extras remain a future provider-specific lane
  only after native provider objects are proven.
