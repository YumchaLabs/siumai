---
name: siumai-ai-sdk-maintenance
description: Maintains Siumai against the Vercel AI SDK reference by auditing public surface parity, core data structures, provider package shape, model catalogs, and new provider onboarding. Use when working in the Siumai Rust repo on AI SDK alignment, provider/model updates, public API compatibility boundaries, or questions about whether Siumai matches upstream AI SDK behavior.
---

# Siumai AI SDK Maintenance

## Quick start

Use this skill only inside the Siumai Rust workspace or while planning Siumai work.

1. Resolve the AI SDK reference repo:
   ```bash
   python .agents/skills/siumai-ai-sdk-maintenance/scripts/resolve_ai_sdk_repo.py
   ```
2. If unresolved, ask for the AI SDK checkout path or set `AI_SDK_REPO`.
3. Read Siumai's local architecture docs before changing code:
   - repository agent instructions supplied by the session or local `AGENTS.md`
   - `docs/architecture/public-surface.md`
   - relevant ADRs under `docs/adr/`
   - active workstream docs if one exists
4. Prefer existing workstream skills for execution: `open-workstream`, `run-workstream-task`,
   `verify-rust-workstream`, and `close-workstream`.

## Reference Repo Resolution

Do not hard-code a user-specific AI SDK path. Resolve in this order:

1. `AI_SDK_REPO`
2. `VERCEL_AI_REPO`
3. nearby `repo-ref/ai`, `repo-ref/vercel-ai`, `../repo-ref/ai`, `../../repo-ref/ai`
4. common sibling names: `ai`, `vercel-ai`, `ai-sdk`

The resolved path must look like the Vercel AI SDK repo: it should include `packages/ai` and at
least one provider package under `packages/`.

## Workflow: Parity Audit

Use for core data structures, interface shape, public exports, or behavior questions.

1. Identify the upstream AI SDK files that define the contract.
2. Identify Siumai owners: spec types, core traits, provider crate, registry, facade exports, docs.
3. Compare structure, names, serde keys, optionality, directionality, and runtime semantics.
4. Classify gaps:
   - `Green`: aligned or intentionally equivalent.
   - `Amber`: acceptable divergence with documented rationale.
   - `Red`: bug, public-surface mismatch, or missing provider behavior.
5. Check compatibility boundaries:
   - `prelude::unified` must stay curated and stable.
   - Legacy compatibility types should remain under explicit compat paths.
   - Provider-specific options/metadata should stay provider-owned.
6. Recommend tests/source guards before implementation.

## Workflow: Model Catalog Refresh

Use when provider model ids, default models, or supported model groups may have drifted.

1. Treat model freshness as time-sensitive. Inspect the AI SDK reference repo and, if the user asks
   for latest/current models, verify against official provider docs or APIs when practical.
2. Find upstream model unions or curated lists in the provider package.
3. Run the local catalog audit before editing when the scope is broader than one provider:
   ```bash
   ./scripts/audit-model-catalogs.sh
   ```
   On Windows, use `scripts\audit-model-catalogs.bat`. The wrapper uses the repository standard
   `--defer deepinfra` gate; call the Python script directly for custom strict audits.
4. Update provider-owned Siumai sources first, usually `siumai-provider-*/src/providers/*/models.rs`.
5. Reuse that source from facade `provider_ext::*::models`, registry catalog, and default-model
   helpers. Avoid duplicated handwritten model arrays.
6. Preserve intentional aliases and hidden compatibility ids only when documented.
7. Add/adjust tests that lock defaults, family grouping, public facade exports, and registry catalog
   output.

## Workflow: New Provider Onboarding

Use when adding or promoting a provider.

1. Decide provider type: native provider, OpenAI-compatible preset, hybrid wrapper, or extension-only.
2. Map AI SDK package surface: provider id, factory functions, model families, settings/options,
   metadata, warnings, and unsupported families.
3. Implement provider-owned identity, config, model catalog, typed options, and metadata extensions.
4. Wire registry factories and facade exports without widening `prelude::unified` unnecessarily.
5. Add public-path tests, lower-contract URL/request tests, capability tests, and docs.
6. Record intentional divergence in docs or ADRs if behavior cannot match AI SDK exactly.

## Validation

Prefer focused commands:

```bash
cargo fmt --check -p <crate>
cargo nextest run -p <crate> <test-filter> --no-fail-fast
cargo check -p <crate> --tests --no-default-features --features <features>
```

If the work is broader than a narrow bug fix, open or reuse a workstream before editing code.
