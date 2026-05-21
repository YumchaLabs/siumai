# FCAB-120 Review — Facade Public Surface Finalization

Date: 2026-05-21
Reviewer: Codex

## Workstream Compliance

- Blocking findings: none.
- The task required finalizing stable prelude, provider extensions, protocol paths, `compat`, and
  `experimental`, while removing or demoting broad mirrors that encourage cross-layer imports.
- `siumai::prelude::unified` now keeps a narrower application-facing helper subset and no longer
  mirrors broad provider-utils helper groups.
- Facade root low-level utility helpers are explicit opt-in imports and are backed by
  `siumai-provider-utils`; `delay` / `is_abort_error` remain the only audited explicit core-runtime
  root helper exception.
- `ToolNameMapping` ownership moved to `siumai-provider-utils::standards`; core keeps only a
  compatibility re-export.
- `experimental` no longer uses the broad grouped `siumai_core` mirror and instead exposes named
  advanced modules.
- `docs/architecture/public-surface.md` now justifies retained broad exports as explicit namespaces:
  protocol, hosted tools, directional content, compatibility prelude, and experimental.

## Code Quality

- Blocking findings: none.
- The facade now points users at owner-backed seams rather than teaching old core utility paths.
- Compile guards prove the public import surface still works after demoting provider-utils helpers
  out of `prelude::unified`.
- Stale tests that implicitly depended on broad legacy `ContentPart` visibility were made explicit
  through compatibility imports, which strengthens the prelude boundary.
- The `siumai-core::standards` compatibility alias keeps migration risk low while provider-utils owns
  the implementation and tests.

## Missing Gates

- No task-local missing gates. Broader workspace smoke remains FCAB-140 scope.

## Residual Risk

- Facade root still exposes many low-level provider-utils helper names for compatibility and opt-in
  utility users. This is acceptable for FCAB-120 because they are explicit root imports, owner-backed,
  and documented as outside the stable unified prelude.
- `delay` / `is_abort_error` still originate from core because they are coupled to cancellation and
  stream runtime wiring; splitting that coupling would be a separate follow-up.

## Verdict

FCAB-120 is ready to mark complete. Proceed to FCAB-130 family taxonomy.
