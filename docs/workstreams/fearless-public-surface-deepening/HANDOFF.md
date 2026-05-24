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

- Task ID: FPSD-070
- Owner: codex
- Files: `siumai-core`, `siumai`, `docs/architecture`, `docs/migration`, source guards,
  workstream evidence.
- Validation: compatibility shim audit document, source guards for retained shims, public compile
  tests for kept compatibility paths.
- Status: READY
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
- Continued FPSD-030 by moving `siumai::experimental` into `siumai/src/experimental.rs` while
  preserving advanced bridge, streaming, execution, provider, and client paths.
- Completed FPSD-030 by moving `siumai::prelude` into `siumai/src/prelude.rs` while preserving
  unified, compatibility, extension, and registry prelude imports.
- Completed FPSD-040 by moving `siumai::image` workflow helpers into
  `siumai/src/image/workflow.rs` and result projection helpers into
  `siumai/src/image/projection.rs`, keeping the root public functions stable.
- Completed FPSD-050 by moving `siumai::video` task workflow helpers into
  `siumai/src/video/workflow.rs`, generated-video extraction/materialization helpers into
  `siumai/src/video/materialization.rs`, and AI SDK result/provider-metadata projection helpers
  into `siumai/src/video/projection.rs`, keeping the root public functions stable.
- Completed FPSD-060 by moving `StreamProcessor` final response assembly into
  `siumai-core/src/streaming/processor/response_assembly.rs`, keeping stream processing behavior and
  provider-map neutrality guarded.

## Blockers

- No blocker currently.

## Next Recommended Action

1. Start FPSD-070 by reading ADR-0007 and ADR-0008, then inventory remaining compatibility shims.
2. Classify each shim as keep, delete now, or split into a future breaking-change lane before
   changing any public compatibility path.
