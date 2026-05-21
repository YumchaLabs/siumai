# FCAB-090 Review — Provider-Utils First Crate Slice

Date: 2026-05-21
Reviewer: Codex

## Workstream Compliance

- Blocking findings: none.
- The task required either a new `siumai-provider-utils` crate or a deep internal provider-utils
  module with a documented split path. The implementation chose the crate path and documented it in
  `docs/architecture/module-split-design.md`.
- The first moved slice is appropriately bounded: URL composition, MIME detection, builder defaults,
  and chat request normalization were already used by provider/protocol code.
- `siumai-core` retains compatibility aliases rather than silently breaking old import paths.
- FCAB-100 remains the right place to classify/delete remaining broad core utility exports.

## Code Quality

- Blocking findings: none.
- The new crate is not merely a pass-through mirror for the moved modules: it owns the implementations
  and tests for the selected helpers.
- The new crate depends on `siumai-spec`, not `siumai-core`, which avoids a dependency cycle and keeps
  the seam lower than runtime core.
- Provider/protocol crates use internal `crate::provider_utils` aliases, so they do not publicly
  mirror the provider-utils crate.
- The `provider_utils_crate_owns_high_churn_provider_helpers` source guard protects against moving
  the selected implementations back into `siumai-core::utils`.

## Missing Gates

- No missing task-local gates. Broader workspace smoke remains a later FCAB-140 concern.

## Residual Risk

- `siumai-core::utils` still has many public re-exports. FCAB-100 should classify them as stable,
  experimental, compat, or provider-utils before removing aliases.
- Some runtime surfaces (`execution`, `streaming`, `retry`, `encoding`) may remain core-stable by
  design; FCAB-100 should apply the deletion test rather than moving them mechanically.

## Verdict

FCAB-090 is ready to mark complete. Run or record final `git diff --check` evidence before moving on.
