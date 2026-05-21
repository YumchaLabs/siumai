# FCAB-070 Review — Directional Content Exports

Date: 2026-05-21

## Workstream Compliance

- Blocking: none.
- Important: none.
- Minor: none.

FCAB-070 satisfies the task contract by separating request prompt content, generated output content,
and legacy compatibility content at the spec/core/facade export seams:

- `siumai-spec::types::content::{prompt, output, compat}`
- `siumai-core::types::content::{prompt, output, compat}`
- `siumai::content::{prompt, output, compat}`
- `siumai::prelude::unified::{prompt, output}` as navigation modules only

Legacy `ContentPart` remains on explicit compatibility paths and is not reintroduced into the
stable unified prelude. The migration and public-surface docs now teach the directional imports.

## Code Quality

- Blocking: none.
- Important: none.
- Minor: none.

The implementation keeps compatibility localized without forcing a wire-format migration in the
same slice. Tests, examples, and `siumai-extras` code that intentionally use legacy content now
import it from `siumai::compat::content`, which makes compatibility usage visible without changing
runtime behavior.

## Missing Gates

None for the FCAB-070 claim. Fresh evidence is recorded in `EVIDENCE_AND_GATES.md` for:

- spec content tests;
- facade/source guards;
- public-surface import guards;
- focused provider/path fixture regressions;
- formatting and diff whitespace checks.

## Residual Risk

- The root spec/core surfaces still retain historical broad exports where ADR 0008 allows them for
  compatibility. FCAB-120 owns final facade narrowing.
- Production protocol/provider/bridge paths still construct legacy `ContentPart` directly in some
  response/request adapters. FCAB-080 owns that migration behind named adapters.

## Verdict

FCAB-070 is ready to remain marked complete. Continue with FCAB-080 and keep the next slice focused
on production direct `ContentPart` construction rather than broader facade cleanup.
