# Fearless Public Surface Deepening — Handoff

Status: Active
Last updated: 2026-05-25

## Current State

This workstream is active as a follow-on to the closed FCAB and module-deepening lanes. It tracks
the next public-surface and runtime deepening queue:

- narrow `siumai::tooling` (done),
- split facade aggregation in `siumai/src/lib.rs`,
- split large `image` and `video` facade Modules,
- perform one or more core streaming deepening slices,
- and audit remaining compatibility shims.

## Active Task

- Task ID: FPSD-030
- Owner: codex
- Files: `siumai/src/lib.rs`, `siumai/src/{hosted_tools,protocol,content,extensions}.rs`,
  facade architecture/public-surface tests, workstream evidence.
- Validation: focused facade architecture gates, `cargo check -p siumai --tests
  --no-default-features --features openai`, and focused public-surface import gates.
- Status: IN PROGRESS. First slice split `hosted_tools`, `protocol`, `content`, and `extensions`
  out of `lib.rs`; `prelude` and `experimental` remain follow-up slices.
- Review: self-review
- Evidence: `EVIDENCE_AND_GATES.md`

## Decisions Since Last Update

- Opened a new active follow-on lane instead of reopening `fearless-clean-architecture-boundaries`,
  because that umbrella lane is closed and its handoff says new work should be concrete follow-ons.
- Chose `FPSD-020` (`siumai::tooling` explicit facade exports) as the first executable task because
  `siumai-core::tooling` was just split and public compile guards already exist.
- Compatibility shim deletion is intentionally last because ADR-0007 and ADR-0008 require migration
  evidence before removing compatibility paths.
- Completed FPSD-010 by opening and committing the workstream docs.
- Completed FPSD-020 by replacing the `siumai::tooling` wildcard mirror with explicit re-exports
  and adding `facade_tooling_module_exports_an_explicit_runtime_surface`.
- Started FPSD-030 by moving four pure public namespace modules out of `siumai/src/lib.rs` while
  keeping the same `siumai::{hosted_tools,protocol,content,extensions}` paths.

## Blockers

- No blocker currently.

## Next Recommended Action

1. Finish and commit the first FPSD-030 namespace split after the package/public-surface gates pass.
2. Continue FPSD-030 by extracting either `experimental` or `prelude` into a named module, keeping
   public paths stable and adding a focused guard.
