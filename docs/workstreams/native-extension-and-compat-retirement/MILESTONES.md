# Native Extension And Compat Retirement — Milestones

Status: Active
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

Exit criteria:

- The lane references ADR-0007 and ADR-0008 explicitly.
- The three requested refactor areas are split into independently verifiable tasks.
- The first executable task is an inventory that can prevent speculative provider edits.

## M1 — Extension Factory Inventory

Exit criteria:

- All `ProviderExtensionFactory` default fallbacks are listed.
- Built-in provider factories with native extension clients are classified.
- The first safe native override target is selected with evidence.

Primary gates:

- source inventory commands recorded in `EVIDENCE_AND_GATES.md`
- existing registry boundary tests selected for follow-up verification

## M2 — Native Extension Factory Convergence

Exit criteria:

- At least one native-capable provider extension path bypasses the generic-client adapter fallback.
- A source guard prevents the same path from regressing into `compat_*_client_with_ctx`.
- No stable family handle stores or depends on `ProviderCompatibilityFactory`.

Primary gates:

- `cargo check -p siumai-registry --tests --no-default-features --features openai`
- focused `factory_architecture_boundary_test` filters

## M3 — Method-Style / Generic-Client Retirement Plan

Exit criteria:

- `ProviderCompatibilityFactory`, `compat_*_client*`, and lower-level core client aliases have
  concrete deletion prerequisites.
- Source guards classify remaining usage as migration-only.
- Migration and architecture docs point users to family-native or extension-native construction.

## M4 — ADR-0008 Root ContentPart Move Preparation

Exit criteria:

- Root `ContentPart` blockers have executable parity gates, not only prose.
- Any namespace movement preserves `ChatMessage` and `ChatResponse` serde behavior.
- Public import tests distinguish stable directional content paths from explicit legacy compat paths.

## M5 — Closeout

Exit criteria:

- Evidence is recorded for all completed slices.
- Remaining provider-specific native overrides or breaking removals are explicit follow-ons.
- `WORKSTREAM.json`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, and `HANDOFF.md` agree.
