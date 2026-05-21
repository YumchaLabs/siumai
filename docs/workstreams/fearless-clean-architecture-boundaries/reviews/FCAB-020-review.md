# FCAB-020 Review

Status: Accepted
Date: 2026-05-21
Scope: Seam inventory and baseline source guards.

## Workstream Compliance

- No blocking findings.
- The task goal is satisfied by `seam-inventory.md`, which records the current registry,
  core/provider-utils, directional content, OpenAI-compatible, bridge, facade, and family-taxonomy
  seams.
- The task validation is satisfied by focused nextest guard runs and formatting/whitespace checks
  recorded in `EVIDENCE_AND_GATES.md`.
- The task stayed inside FCAB-020 scope: docs plus source guards only. No production API or runtime
  behavior was changed.

## Code Quality

- No blocking findings.
- The new registry guard is intentionally narrow: it protects the known regression where generic
  `LlmClient` factory methods could silently become primary unprefixed construction paths again.
- The new facade guard keeps the FCAB seam inventory discoverable from the existing architecture
  guard suite without weakening older `ContentPart` audits.
- The source guards are string-based, matching existing architecture-test style in this repository.

## Missing Gates

- No missing gates for FCAB-020.
- Broader package gates remain intentionally deferred to the implementation slices that move code.

## Residual Risk

- The inventory is a baseline, not a final architecture. FCAB-030/040 still need to actually split
  `ProviderFactory` and legacy `LlmClient` construction.
- FCAB-090 still needs an implementation-time decision between a new provider-utils crate and a deep
  internal provider-utils module.
- `$verify-rust-workstream` should run before final lane closeout, not for this source-guard-only
  baseline slice.
