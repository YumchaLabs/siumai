# Fearless Public Surface Deepening — TODO

Status: Active
Last updated: 2026-05-25

Status legend:

- `[ ]` pending
- `[~]` in progress
- `[x]` complete
- `[-]` intentionally deferred or split

## M0 — Scope And Evidence Freeze

- [x] FPSD-010 [owner=planner] [deps=none] [scope=docs/workstreams/fearless-public-surface-deepening]
  Goal: Freeze problem, target state, non-goals, task order, and evidence anchors for the public
  surface deepening lane.
  Validation: `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`,
  `WORKSTREAM.json`, and `HANDOFF.md` exist and agree.
  Review: planner self-review.
  Evidence: workstream docs.
  Handoff: DONE. FPSD-020 is the first executable coding task.

## M1 — Facade Surface Tightening

- [x] FPSD-020 [owner=codex] [deps=FPSD-010] [scope=siumai/src/tooling.rs,siumai/tests,docs/architecture,docs/migration]
  Goal: Narrow `siumai::tooling` to an explicit curated re-export surface over
  `siumai_core::tooling` after the core tooling implementation split.
  Validation:
  - `cargo check -p siumai --tests --no-default-features --features openai`
  - `cargo nextest run -p siumai --test tooling_runtime_public_surface_test public_surface_tooling_runtime_contract_compiles --no-default-features --features openai --no-fail-fast`
  - `cargo nextest run -p siumai --test public_surface_imports_test public_surface_tooling_imports_compile --no-default-features --features openai --no-fail-fast`
  Review: self-review plus source guard update.
  Evidence: facade explicit export source and public compile guards.
  Handoff: DONE. `siumai::tooling` no longer wildcard-mirrors core and is guarded by facade
  architecture tests. FPSD-030 can split `siumai/src/lib.rs`.

- [x] FPSD-030 [owner=codex] [deps=FPSD-020] [scope=siumai/src/lib.rs,siumai/src/*.rs,siumai/tests,docs/architecture]
  Goal: Split `siumai/src/lib.rs` policy-heavy facade sections into named modules while preserving
  current public paths and prelude behavior.
  Validation:
  - `cargo check -p siumai --tests --no-default-features --features openai`
  - focused facade architecture and public-surface import guards.
  Review: source review for public path preservation.
  Evidence: named facade modules and root export guards.
  Handoff: DONE. Split `hosted_tools`, `protocol`, `content`, `extensions`, `experimental`, and
  `prelude` out of `lib.rs`; large image/video facade Modules can be split next.

## M2 — Large Facade Module Deepening

- [x] FPSD-040 [owner=codex] [deps=FPSD-030] [scope=siumai/src/image.rs,siumai/src/image/**,siumai/tests,docs]
  Goal: Split `siumai::image` implementation by public workflow or helper role while preserving the
  stable `siumai::image::*` surface.
  Validation:
  - focused image facade compile/runtime tests selected from current public surface coverage.
  Review: verify no provider-owned runtime logic moves into the facade.
  Evidence: image module split guard and public import coverage.
  Handoff: DONE. Split image workflow and projection helpers into named private submodules while
  keeping `siumai::image::*` stable; FPSD-050 can reuse the same root-plus-helper pattern for video.

- [x] FPSD-050 [owner=codex] [deps=FPSD-030] [scope=siumai/src/video.rs,siumai/src/video/**,siumai/tests,docs]
  Goal: Split `siumai::video` implementation by request/result/materialization helpers while
  preserving the stable `siumai::video::*` surface.
  Validation:
  - focused video facade and family import tests.
  Review: verify task-oriented video family semantics remain unchanged.
  Evidence: video module split guard and public import coverage.
  Handoff: DONE. Split video workflow, materialization, and projection helpers into named private
  submodules while keeping `siumai::video::*` stable; FPSD-060 can move into core streaming.

## M3 — Core Streaming Deepening

- [x] FPSD-060 [owner=codex] [deps=FPSD-020] [scope=siumai-core/src/streaming,siumai-core/tests,docs]
  Goal: Split one high-value core streaming Module slice into named submodules without changing
  stream behavior or provider-map neutrality.
  Validation:
  - `cargo check -p siumai-core --tests --no-default-features`
  - focused streaming tests for the touched Module.
  Review: stronger review required because this is runtime behavior surface.
  Evidence: stream split source guard and focused nextest results.
  Handoff: DONE. Split `StreamProcessor` final response assembly into
  `siumai-core/src/streaming/processor/response_assembly.rs` while preserving stream behavior and
  provider-map neutrality; FPSD-070 can audit compatibility shims next.

## M4 — Compatibility Shim Audit

- [x] FPSD-070 [owner=codex] [deps=FPSD-020] [scope=siumai-core,siumai,docs/architecture,docs/migration]
  Goal: Audit remaining compatibility shims and classify each as keep, delete now, or split into a
  future breaking-change lane.
  Validation:
  - source guards for retained shims;
  - public compile tests for kept compatibility paths;
  - migration docs for any removal.
  Review: check ADR-0007 and ADR-0008 before deleting.
  Evidence: compatibility shim audit document and tests.
  Handoff: DONE. Added `compatibility-shim-audit.md`; no remaining shim is safe to delete in this
  lane because ADR-0007 / ADR-0008 and migration docs require explicit compatibility retention.
  Future removals are split to breaking-change follow-up candidates.

## M5 — Integration And Closeout

- [ ] FPSD-080 [owner=planner] [deps=FPSD-020,FPSD-030,FPSD-040,FPSD-050,FPSD-060,FPSD-070] [scope=docs/workstreams/fearless-public-surface-deepening]
  Goal: Close the lane or split unresolved work into narrower follow-ons.
  Validation:
  - documented final gate matrix in `EVIDENCE_AND_GATES.md`
  - `git diff --check -- docs/workstreams/fearless-public-surface-deepening`
  Review: final self-review or `review-workstream`.
  Evidence: updated `WORKSTREAM.json`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, and
  `HANDOFF.md`.
  Handoff: Summarize residual risks and next lane if any.
