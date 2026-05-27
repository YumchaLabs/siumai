# Fearless Residual Architecture Deepening - Milestones

Status: Active
Last updated: 2026-05-27

## M0 - Planning

Exit criteria:

- Workstream docs exist and agree on task order.
- `WORKSTREAM.json` parses.
- The task ledger covers all five architecture-review candidates.

## M1 - Registry Provider Descriptor Seam

Exit criteria:

- Provider metadata, aliases, defaults, model catalog, built-in factory selection, and catalog views
  cross one registry-owned descriptor seam.
- Existing provider catalog and factory architecture tests pass.
- Repeated provider facts are deleted or demoted to compatibility-only adapters.

## M2 - Protocol And Bridge Conversion Seams

Exit criteria:

- OpenAI-compatible message dialect conversion is split from broad utility helpers.
- Bridge request normalization routes each supported wire format through a codec Module.
- Protocol and bridge fixtures prove no request/response conversion regression.

## M3 - Test Harness Depth

Exit criteria:

- Provider contract/public-path parity tests use scenario/harness seams instead of manual matrices
  where practical.
- Coverage remains equivalent or stronger for modified provider families.
- Source guards remain focused on architectural invariants, not duplicated behavior fixtures.

## M4 - ADR-0008 ContentPart Decision Or Move

Exit criteria:

- ADR-0008 root-move gates are proven and the move lands, or a concrete gate blocker is recorded
  with an executable test path.
- Compatibility serde stays stable for `ChatMessage` and `ChatResponse`.
- New request/response code paths do not learn the root `ContentPart` path.

## M5 - Closeout

Exit criteria:

- Fresh targeted gates pass for all touched crates.
- Workstream docs, changelogs, and evidence agree on final state.
- Any follow-on is split only when it has a narrower scope and concrete evidence.
