# Fearless Public Surface Deepening — Handoff

Status: Closed
Last updated: 2026-05-25

## Current State

This workstream is closed as a follow-on to the closed FCAB and module-deepening lanes. It completed
the public-surface and runtime deepening queue:

- narrow `siumai::tooling` (done),
- split facade aggregation in `siumai/src/lib.rs`,
- split large `image` and `video` facade Modules,
- perform one or more core streaming deepening slices,
- and audit remaining compatibility shims.

## Active Task

None. The lane is closed.

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
- Completed FPSD-070 by adding `compatibility-shim-audit.md`, classifying remaining compatibility
  shims against ADR-0007 and ADR-0008, adding an audit coverage guard, and extending public compat
  import coverage for retained client/builder/content/type paths. No shim is safe to delete in this
  lane; future removals are breaking-lane candidates.
- Completed FPSD-080 by recording final evidence, marking the lane closed, and leaving future
  compatibility removals as explicit breaking-lane candidates rather than open work in this lane.

## Blockers

- No blocker currently.

## Follow-On Candidates

1. Open a narrow compatibility-breaking lane for method-style facade construction removal when
   public examples and downstream migration no longer require `siumai::compat::{Siumai,
   SiumaiBuilder, Provider}`.
2. Open a core compatibility alias removal lane for `siumai_core::client` and
   `siumai_core::core::client` after generic-client users migrate to `siumai_core::compat::client`.
3. Open a broad compat namespace narrowing lane for `siumai::compat::types::*` and
   `siumai::prelude::compat::types::*`.
4. Open a future ADR-0008 breaking slice for legacy `ContentPart` namespace movement once
   directional adapters and fixture parity satisfy the ADR conditions.
5. Open a registry compatibility-factory retirement lane after method-style `Siumai` and
   extension-only generic-client adapters have native family or extension factories.
