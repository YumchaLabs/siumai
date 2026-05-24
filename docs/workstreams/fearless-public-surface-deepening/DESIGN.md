# Fearless Public Surface Deepening

Status: Closed
Last updated: 2026-05-25

## Why This Lane Exists

The 2026-05-21 clean-architecture lane closed the largest cross-crate boundary pass. The next
remaining friction is narrower: several facade and runtime Modules still have shallow Interfaces or
large mixed-purpose Implementation files. They are not incorrect, but they make future provider,
streaming, compatibility, and public-surface changes harder to review.

## Relevant Authority

- ADRs:
  - `docs/adr/0001-vercel-aligned-modular-split.md`
  - `docs/adr/0006-family-model-first-trait-policy.md`
  - `docs/adr/0007-llmclient-demotion-policy.md`
  - `docs/adr/0008-legacy-content-part-compatibility-boundary.md`
- Existing docs:
  - `docs/architecture/public-surface.md`
  - `docs/architecture/module-split-design.md`
  - `docs/workstreams/fearless-clean-architecture-boundaries/HANDOFF.md`
  - `docs/workstreams/fearless-clean-architecture-boundaries/provider-utils-classification.md`
- Related workstreams:
  - `docs/workstreams/fearless-clean-architecture-boundaries/`
  - `docs/workstreams/fearless-module-deepening/`
  - `docs/workstreams/provider-utils-tooling-runtime-alignment/`
  - `docs/workstreams/ai-sdk-structural-alignment/`

## Problem

The public facade and core runtime still contain shallow or oversized Modules:

- `siumai/src/tooling.rs` mirrors `siumai_core::tooling::*` with a wildcard export, unlike the
  already-narrowed `ui` and `retry_api` facades.
- `siumai/src/lib.rs` mixes root exports, prelude policy, compatibility paths, experimental modules,
  content namespaces, protocol helpers, and hosted-tool surfaces in one file.
- `siumai/src/image.rs` and `siumai/src/video.rs` are large facade Modules where helper execution,
  request/result projection, compatibility aliases, and public imports are hard to navigate.
- `siumai-core/src/streaming/{stream_part,processor}.rs` remain high-value but large runtime
  Modules.
- Compatibility shims such as `siumai-core::client`, `siumai-core::core::client`,
  `siumai-core::standards::tool_name_mapping`, `siumai::compat::*`, and root
  `delay` / `is_abort_error` need a fresh deletion or retention audit.

## Target State

- Facade Modules expose explicit curated re-exports instead of wildcard mirrors when the public
  contract is known.
- Large facade Modules are split by public workflow while preserving existing import paths.
- Core streaming Modules are deepened behind named submodules without weakening stream behavior or
  provider-map neutrality guards.
- Compatibility shims are either retained with documented reasons and guards, or removed only when
  migration docs and public compile guards prove the break is intentional.

## In Scope

- Narrow `siumai::tooling` to explicit re-exports.
- Split `siumai/src/lib.rs` internals into named facade modules while preserving public paths.
- Split `siumai/src/image.rs` and `siumai/src/video.rs` into named implementation modules.
- Split core streaming modules where a small, behavior-preserving slice has clear test coverage.
- Audit remaining compatibility shims and open a compatibility-break follow-on if removal requires
  a broader public contract decision.

## Out Of Scope

- Changing provider runtime semantics.
- Reopening the closed FCAB umbrella lane.
- Removing compatibility APIs without migration documentation and public-surface compile coverage.
- Promoting Music to a stable family; ADR-0006 requires a separate ADR for that.
- Broad workspace formatting or smoke scripts that are known to be unreliable in this Windows
  terminal.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Explicit facade re-exports improve reviewability without breaking callers when the import surface is covered by compile tests. | High | `siumai::ui` and `siumai::retry_api` were already narrowed and guarded. | Revert to a compatibility wildcard only for the affected module and document why. |
| `siumai/src/lib.rs`, `image.rs`, and `video.rs` can be split without public API changes. | Medium | Rust module roots can re-export from child modules; existing public tests cover many imports. | Keep the root file as the public surface and split only private helpers. |
| Core streaming split requires stronger evidence than facade splits. | High | Streaming is a core behavior contract with broad provider impact. | Use smaller slices and run focused stream package gates before committing. |
| Compatibility shim deletion may need a separate breaking-change lane. | High | ADR-0007 and ADR-0008 keep compatibility surfaces available during migration. | Record retention reasons instead of deleting in this lane. |

## Architecture Direction

The lane follows the same Module-deepening pattern already used for `siumai-core::ui` and
`siumai-core::tooling`: preserve the public Interface at the Module root, move Implementation
details into named submodules, and add source guards that prevent collapsing back into shallow
wildcard or monolithic roots.

Public-facing facade work should prefer explicit imports and curated prelude surfaces. Core runtime
work should preserve provider-agnostic ownership and avoid introducing protocol/provider knowledge
into stable core Modules.

## Closeout Condition

This lane can close when:

- the `tooling` facade is explicit and guarded,
- the facade aggregation and large image/video Modules are either split or explicitly deferred with
  reasons,
- at least one core streaming deepening slice is completed or split into a narrower child lane,
- compatibility shims are audited with keep/delete decisions,
- evidence gates pass, and
- follow-on work is either completed, deferred, or split into a new workstream.

## Closeout Result

Closed on 2026-05-25.

The lane met its target state:

- `siumai::tooling` now exposes a curated facade surface instead of wildcard-mirroring core.
- `siumai/src/lib.rs` no longer owns the large hosted-tools, protocol, content, extensions,
  experimental, or prelude bodies inline.
- `siumai::image` and `siumai::video` now keep stable public roots over named helper modules.
- `StreamProcessor` final response assembly moved behind a named core streaming helper module.
- Remaining compatibility shims are classified in `compatibility-shim-audit.md`.

No compatibility shim was deleted in the final audit slice because the remaining shims are either
explicit migration namespaces or future breaking-lane candidates governed by ADR-0007 / ADR-0008.
