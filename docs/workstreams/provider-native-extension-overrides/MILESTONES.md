# Provider Native Extension Overrides — Milestones

Status: Active
Last updated: 2026-05-25

## M0 — Scope And Inventory

Exit criteria:

- Workstream docs exist and agree on active task PNEO-010/PNEO-020.
- `docs/workstreams/INDEX.md` reflects the actual workstream inventory.
- Selected provider-native extension candidates are recorded.

## M1 — Native Extension Hook Overrides

Exit criteria:

- Azure OpenAI, OpenAI, Anthropic, Gemini, xAI, and MiniMaxi file hooks return native provider
  clients directly where supported.
- OpenAI and Anthropic skills hooks return native provider clients directly.
- MiniMaxi music hook returns the native provider client directly.
- Source guards reject fallback through `compat_language_client_with_ctx(...)`,
  `as_*_capability()`, and `ClientBacked*` adapters for those hooks.

## M2 — Evidence And Closeout

Exit criteria:

- Focused nextest and check gates pass for the touched registry feature set.
- Residual extension defaults are documented.
- The lane is closed or has a concrete follow-on instead of open-ended active scope.
