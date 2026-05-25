# Compatibility Surface Breaking Convergence — TODO

Status: Active
Last updated: 2026-05-25

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[-]` intentionally deferred or split

## M0 — Scope And Evidence Freeze

- [x] CSBC-010 [owner=planner] [deps=none] [scope=docs/workstreams/compatibility-surface-breaking-convergence]
  Goal: Freeze target state, task order, and evidence anchors for compatibility-surface breaking
  convergence.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree.
  Review: planner self-review.
  Evidence: workstream docs.
  Handoff: DONE. Workstream docs opened and indexed. First executable task is CSBC-020.

## M1 — Broad Facade Compat Types Narrowing

- [x] CSBC-020 [owner=codex] [deps=CSBC-010] [scope=siumai/src/compat.rs,siumai/src/prelude.rs,siumai/tests,docs]
  Goal: Replace broad `siumai::compat::types::*` / `siumai::prelude::compat::types::*` mirroring
  with a narrower explicit compatibility type surface or a documented transition module.
  Validation:
  - `cargo check -p siumai --tests --no-default-features --features openai`
  - focused public compat import tests.
  Review: public path preservation and migration-doc review.
  Evidence: facade source guard and public import coverage.
  Handoff: DONE. `compat::types` and `prelude::compat::types` now expose common legacy names only;
  the old catch-all mirror moved to nested `legacy_all` modules for last-resort migration.

## M2 — Core Generic Client Alias Exit Preparation

- [x] CSBC-030 [owner=codex] [deps=CSBC-020] [scope=siumai-core/src,siumai-core/tests,docs/migration,docs/architecture]
  Goal: Move safe internal/test usage away from `siumai_core::client` /
  `siumai_core::core::client`, strengthen guards, and define removal criteria for the aliases.
  Validation:
  - `cargo check -p siumai-core --tests --no-default-features`
  - focused `core_provider_boundary_test` filters.
  Review: ADR-0007 compliance review.
  Evidence: source guard updates and migration docs.
  Handoff: DONE. `siumai_core::client` and `siumai_core::core::client` are deprecated migration
  aliases with ADR-0007 removal guidance; guards now prevent production code from consuming those
  aliases as implementation owners.

## M3 — Registry Generic-Client Factory Seam Reduction

- [x] CSBC-040 [owner=codex] [deps=CSBC-030] [scope=siumai-registry/src,siumai-registry/tests,docs]
  Goal: Reduce stable-family dependency on `ProviderCompatibilityFactory` /
  `compat_*_client*` and document remaining extension/method-style dependency points.
  Validation:
  - `cargo check -p siumai-registry --tests --no-default-features --features openai`
  - focused registry factory architecture tests.
  Review: ADR-0007 and family-first registry review.
  Evidence: registry source guards and architecture docs.
  Handoff: DONE. Image, speech, and transcription extras now route through
  `ProviderExtensionFactory`; stable family handles no longer store `ProviderCompatibilityFactory`.
  `ProviderFactoryFacets` keeps only family/extension facets, while the compatibility facet is
  created only for explicit `SiumaiBuilder` generic-client migration construction.

## M4 — ADR-0008 ContentPart Breaking-Slice Decision

- [x] CSBC-050 [owner=codex] [deps=CSBC-020] [scope=siumai-core,siumai-spec,siumai,docs/adr,docs/migration,tests]
  Goal: Evaluate ADR-0008 future-breaking conditions and either execute a safe compatibility
  namespace break or record the exact blockers with source guards.
  Validation:
  - focused content boundary tests;
  - public content import tests;
  - fixture parity tests if any public path moves.
  Review: ADR-0008 compliance review.
  Evidence: decision note and tests.
  Handoff: DONE_WITH_CONCERNS. The facade-level break is already complete:
  `prelude::unified` does not export legacy `ContentPart`, and explicit compat content namespaces
  exist. A full `siumai-spec::types::ContentPart` / `siumai-core::types::ContentPart` root move is
  deferred because serde-facing `ChatMessage` / `ChatResponse`, provider/protocol response parity,
  and a full root-move fixture suite are not yet complete. The blockers are recorded in
  `CSBC-050-content-part-decision.md` and guarded by
  `adr_0008_full_contentpart_namespace_break_blockers_are_guarded`.

## M5 — Closeout

- [ ] CSBC-060 [owner=planner] [deps=CSBC-020,CSBC-030,CSBC-040,CSBC-050] [scope=docs/workstreams/compatibility-surface-breaking-convergence]
  Goal: Close this lane or split remaining compatibility removals into narrower follow-ons.
  Validation:
  - documented final gate matrix in `EVIDENCE_AND_GATES.md`
  - `git diff --check -- docs/workstreams/compatibility-surface-breaking-convergence`
  Review: final self-review or `review-workstream`.
  Evidence: updated workstream docs.
  Handoff: Summarize residual public API risks.
