# Provider Model Catalog AI SDK Refresh — Milestones

Status: Closed
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

Status: Complete.

Exit criteria:

- Low-risk providers and DeepInfra deferral are explicit.
- Audit baseline is recorded.
- First implementation slice is chosen.

Primary evidence:

- `docs/workstreams/provider-model-catalog-ai-sdk-refresh/DESIGN.md`
- `docs/workstreams/provider-model-catalog-ai-sdk-refresh/TODO.md`

## M1 — Low-Risk Catalog Refresh

Status: Complete.

Exit criteria:

- Alibaba/Qwen, Anthropic, Cohere, Google/Gemini, Google Vertex, Mistral, and xAI no longer report missing upstream ids in the local audit.
- Compatibility aliases remain intact.
- Registry/facade paths still reuse provider-owned constants.

Primary gates:

- `python .agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py --include-green --show-skipped --defer deepinfra`

## M2 — Tests And Gates

Status: Complete.

Exit criteria:

- Focused tests prove refreshed ids are visible in relevant model sets/catalogs.
- Formatting is clean for touched packages.

Primary gates:

- `cargo fmt --check` scoped to touched packages where practical.
- Focused `cargo nextest` commands for edited provider/registry tests.

## M3 — Closeout

Status: Complete.

Exit criteria:

- Gate evidence is recorded.
- DeepInfra policy work is either completed or explicitly deferred.
- `WORKSTREAM.json` status reflects the final state.
