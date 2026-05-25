# Compatibility Surface Breaking Convergence — Milestones

Status: Closed
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

Exit criteria:

- Requirements are split into compatibility-surface tasks.
- ADR-0007 and ADR-0008 are explicitly referenced.
- First executable slice is low-risk and independently verifiable.

## M1 — Broad Facade Compat Types Narrowing

Exit criteria:

- `siumai::compat::types` no longer blindly mirrors every `siumai_core::types` name, or a
  transition module is explicitly documented and guarded.
- `prelude::compat::types` follows the same narrowed surface.
- Public import tests prove retained migration paths.

Primary gates:

- `cargo check -p siumai --tests --no-default-features --features openai`
- focused facade architecture and public-surface import tests.

## M2 — Core Generic Client Alias Exit Preparation

Exit criteria:

- New core and registry code is guarded toward `compat::client` imports.
- Any retained `siumai_core::client` / `siumai_core::core::client` aliases have explicit removal
  conditions.
- No stable family path depends on the aliases as implementation owners.

## M3 — Registry Generic-Client Factory Seam Reduction

Exit criteria:

- Stable family handles remain native-family first.
- Stable family handles store only family/extension facets, not `ProviderCompatibilityFactory`.
- Remaining compatibility factory methods are documented as explicit method-style migration seams.
- Guards prevent stable family regressions into `compat_*_client*` self-calls or direct
  compatibility-facet storage.

## M4 — ADR-0008 ContentPart Breaking-Slice Decision

Exit criteria:

- ADR-0008 future-breaking conditions are checked against current source.
- The completed facade-level compatibility break is distinguished from the blocked low-level
  spec/core root namespace move.
- Blockers for the full root move are recorded as concrete follow-on prerequisites and guarded by
  source tests.

## M5 — Closeout

Exit criteria:

- Evidence is recorded for all completed slices.
- Remaining breaking changes are explicit, not hidden.
- `WORKSTREAM.json`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, and `HANDOFF.md` agree.

Result: complete. Remaining breaking changes are follow-on candidates:

- delete lower-level core generic-client aliases after ADR-0007 conditions are met;
- retire method-style/generic-client construction after family-native providers and extension
  factories cover those use cases;
- move low-level `ContentPart` root paths only after ADR-0008 root-move parity prerequisites are
  satisfied.
