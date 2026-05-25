# Provider Model Catalog AI SDK Refresh — TODO

Status: Closed
Last updated: 2026-05-25

## M0 — Scope And Evidence Freeze

- [x] PMCA-010 [owner=planner] [deps=none] [scope=docs/workstreams/provider-model-catalog-ai-sdk-refresh]
  Goal: Freeze the low-risk model catalog refresh scope and explicitly defer DeepInfra policy work.
  Validation: DESIGN.md, TODO.md, MILESTONES.md, EVIDENCE_AND_GATES.md, WORKSTREAM.json exist and agree.
  Evidence: docs/workstreams/provider-model-catalog-ai-sdk-refresh/DESIGN.md
  Handoff: Complete; implementation may proceed.

## M1 — Low-Risk Catalog Refresh

- [x] PMCA-020 [owner=codex] [deps=PMCA-010] [scope=siumai-provider-*,siumai/src/provider_ext,siumai-registry]
  Goal: Add missing AI SDK model ids for Alibaba/Qwen, Anthropic, Cohere, Google/Gemini, Google Vertex, Mistral, and xAI without removing compatibility aliases.
  Validation: `python .agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py --provider <target>` for each touched target.
  Review: Check that constants remain provider-owned and reused by facade/registry paths.
  Evidence: `.agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py`
  Handoff: Complete. The full audit is green for all low-risk targets with DeepInfra explicitly deferred.

## M2 — Tests And Gates

- [x] PMCA-030 [owner=codex] [deps=PMCA-020] [scope=edited provider crates,registry]
  Goal: Add or update focused tests that prove refreshed ids are exposed by model sets and catalogs.
  Validation: focused `cargo nextest` commands for edited crates/tests plus `cargo fmt --check` for touched packages.
  Review: Ensure tests lock public catalog visibility instead of only checking non-empty constants.
  Evidence: EVIDENCE_AND_GATES.md
  Handoff: Complete. Focused provider, registry, and public-surface gates passed.

## M3 — Closeout

- [x] PMCA-040 [owner=planner] [deps=PMCA-030] [scope=docs/workstreams/provider-model-catalog-ai-sdk-refresh]
  Goal: Close the lane or split DeepInfra into a narrower follow-on.
  Validation: final audit and fresh targeted Rust gates are recorded.
  Review: No blocking findings remain.
  Evidence: EVIDENCE_AND_GATES.md, WORKSTREAM.json
  Handoff: Closed. DeepInfra remains a separate policy follow-on if full upstream catalog parity is desired.
