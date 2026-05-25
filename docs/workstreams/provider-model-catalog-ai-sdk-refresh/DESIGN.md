# Provider Model Catalog AI SDK Refresh

Status: Closed
Last updated: 2026-05-25

## Why This Lane Exists

Siumai exposes curated model constants and registry catalogs for provider surfaces that intentionally track the Vercel AI SDK package model-id unions. A local audit against `repo-ref/ai` found several provider catalogs that are stale even though the underlying providers already support arbitrary model id strings at runtime.

## Relevant Authority

- Local skill: `.agents/skills/siumai-ai-sdk-maintenance/SKILL.md`
- Audit script: `.agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py`
- Related workstreams:
  - `docs/workstreams/ai-sdk-structural-alignment/`
  - `docs/workstreams/openai-compatible-package-surface-alignment/`
  - `docs/workstreams/google-package-surface-alignment/`
  - `docs/workstreams/google-vertex-package-surface-alignment/`
  - `docs/workstreams/xai-package-surface-alignment/`
  - `docs/workstreams/deepinfra-unified-provider-surface/`

## Problem

Provider catalog constants claim AI SDK alignment but miss upstream model ids for Alibaba, Anthropic, Cohere, Google/Gemini, Google Vertex, Mistral, and xAI. DeepInfra also has a large mismatch, but its curated-subset policy needs a separate decision before bulk expansion.

## Target State

The low-risk provider catalogs cover the current local AI SDK model-id unions, intentional legacy aliases remain available, and the audit script reports only the explicitly deferred DeepInfra policy gap.

Closeout: Achieved on 2026-05-25. The local audit reports no red drift outside the documented DeepInfra deferral, and focused Rust gates cover provider-owned model sets, registry catalog visibility, and public facade imports.

## In Scope

- Add missing AI SDK model ids for Alibaba/Qwen, Anthropic, Cohere, Google/Gemini, Google Vertex, Mistral, and xAI.
- Keep provider-owned constants as the source of truth for registry/facade reuse.
- Add focused tests or source guards that make the refreshed ids visible in public catalogs.
- Record DeepInfra as deferred policy work instead of silently mixing it into this small refresh.

## Out Of Scope

- Dependency vulnerability audit for release.
- Official provider-doc freshness verification beyond the local AI SDK checkout.
- Removing old model aliases that may still be compatibility surface.
- DeepInfra full catalog expansion.

## Starting Assumptions

| Assumption | Confidence | Evidence | Consequence if wrong |
| --- | --- | --- | --- |
| Local `repo-ref/ai` is the intended AI SDK baseline for this refresh. | High | User stated behavior generally follows Vercel AI SDK; resolver found `repo-ref/ai`. | If stale, a later official-doc refresh may add or remove ids. |
| Missing model constants are low-risk because runtime APIs accept arbitrary strings. | High | Builders and request types pass model strings through. | If some ids need behavior-specific mapping, tests should expose it. |
| DeepInfra needs a policy decision before bulk update. | Medium | Audit found 51 missing ids while existing docs say curated subset. | If full parity is required, open a dedicated DeepInfra catalog task. |

## Architecture Direction

Provider-owned model files should remain authoritative. Registry and facade exports should reuse those constants instead of duplicating handwritten arrays. The audit script is the repeatable maintenance gate for comparing upstream TypeScript model-id unions with Siumai Rust constants.

## Closeout Condition

This lane can close when:

- low-risk provider model catalog drift is fixed,
- the audit script reports no red targets except documented DeepInfra deferral,
- focused Rust tests pass for edited crates,
- evidence is recorded in `EVIDENCE_AND_GATES.md`,
- and DeepInfra follow-on scope is explicitly captured.
