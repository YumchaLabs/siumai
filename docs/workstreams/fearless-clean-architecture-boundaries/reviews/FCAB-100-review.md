# FCAB-100 Review — Provider-Utils Deepening And Classification

Date: 2026-05-21
Reviewer: Codex

## Workstream Compliance

- Blocking findings: none.
- The task required protocol/provider utility imports to move to the provider-utils seam. The last
  stale TogetherAI `siumai_core::utils::builder_helpers` import now uses
  `crate::provider_utils::builder_helpers`.
- The task required remaining `siumai-core::utils` public exports to be classified. The
  classification is recorded in `provider-utils-classification.md` and enforced by
  `core_utils_remaining_owned_modules_are_classified`.
- The implementation respects the deletion test from `DESIGN.md`: spec-only AI SDK-style helpers
  moved to `siumai-provider-utils`; core stream/cancellation implementations stayed in core instead
  of creating a dependency cycle or hollow abstraction.
- FCAB-110 bridge-target work and FCAB-120 facade tightening were not pulled into this slice.

## Code Quality

- Blocking findings: none.
- `siumai-provider-utils` remains lower than `siumai-core`: it imports `siumai-spec` and generic
  runtime libraries, not core runtime modules.
- `siumai-core::utils` compatibility aliases are shallow by design, while the real implementations
  and tests live in `siumai-provider-utils`.
- Provider/protocol/registry import hygiene is guarded by
  `provider_protocol_crates_do_not_import_moved_provider_utils_from_core`.
- Keeping `cancel` in core is correct because its public functions expose `ChatStream`,
  `ChatStreamHandle`, and `CancelHandle`. Keeping `streaming_tool_call` in core compat is also
  correct until stream part ownership is revisited because it constructs core
  `LanguageModelV4StreamPart` values.

## Missing Gates

- No task-local missing gates after the focused provider-utils/core/provider checks in
  `EVIDENCE_AND_GATES.md`.
- Broader all-workspace smoke remains a later FCAB-140 closeout gate.

## Residual Risk

- Facade root low-level utility imports remain for advanced utility users and are documented as
  explicit imports. FCAB-120 should decide whether any of those aliases should be further demoted to
  `experimental` or `compat`.
- Provider crates still use `crate::utils::cancel` for core cancellation helpers by design; future
  work should not treat that as provider-utils leakage.

## Verdict

FCAB-100 is ready to mark complete after final validation evidence is recorded.
