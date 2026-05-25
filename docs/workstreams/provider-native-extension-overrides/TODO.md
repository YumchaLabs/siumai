# Provider Native Extension Overrides — TODO

Status: Active
Last updated: 2026-05-25

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[-]` intentionally deferred or split

## M0 — Scope And Inventory

- [~] PNEO-010 [owner=planner] [deps=none] [scope=docs/workstreams/provider-native-extension-overrides]
  Goal: Open the follow-on lane for provider-native file, skills, and music extension factory
  overrides.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree; `docs/workstreams/INDEX.md` includes the
  lane.
  Review: planner self-review.
  Evidence: workstream docs.
  Handoff: In progress. First executable code task is PNEO-020.

- [ ] PNEO-020 [owner=codex] [deps=PNEO-010] [scope=siumai-registry/src,siumai-registry/tests]
  Goal: Add a focused source guard that names each selected provider-native extension hook and
  rejects generic-client adapter fallback.
  Validation:
  `cargo nextest run -p siumai-registry --test factory_architecture_boundary_test provider_native_extension_hooks_bypass_generic_client_adapters --no-default-features --features openai,azure,anthropic,google,xai,minimaxi --no-fail-fast`
  Review: registry construction-boundary review.
  Evidence: failing-then-passing boundary test.
  Handoff: PNEO-030 should satisfy the guard without broad provider rewrites.

## M1 — Native Extension Hook Overrides

- [ ] PNEO-030 [owner=codex] [deps=PNEO-020] [scope=siumai-registry/src/registry/factories]
  Goal: Override provider-native file, skills, and music extension hooks where the typed provider
  client already implements the target extension trait.
  Validation:
  `cargo check -p siumai-registry --tests --no-default-features --features openai,azure,anthropic,google,xai,minimaxi`
  Review: provider factory review.
  Evidence: typed-client projection in factory methods.
  Handoff: PNEO-040 records the final gate matrix and residual extension defaults.

## M2 — Evidence And Closeout

- [ ] PNEO-040 [owner=codex] [deps=PNEO-020,PNEO-030] [scope=docs/workstreams/provider-native-extension-overrides]
  Goal: Record validation evidence, classify remaining defaults, and decide whether the lane can
  close or should split a follow-on.
  Validation:
  - `git diff --check -- docs/workstreams/provider-native-extension-overrides docs/workstreams/INDEX.md siumai-registry/src/registry/factories siumai-registry/tests/factory_architecture_boundary_test.rs`
  - final focused cargo gates from `EVIDENCE_AND_GATES.md`
  Review: final self-review.
  Evidence: updated workstream docs.
  Handoff: close lane if gates pass and residual work is split.
