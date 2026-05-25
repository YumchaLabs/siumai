# Provider Model Catalog AI SDK Refresh — Evidence And Gates

Status: Closed
Last updated: 2026-05-25

## Smallest Current Repro

```powershell
python .agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py --include-green --show-skipped --defer deepinfra
```

Initial result: red drift in Alibaba, Anthropic, Cohere, DeepInfra, Google, Google Vertex, Mistral, and xAI. DeepInfra is deferred because it has a large curated-subset policy question.

## Gate Set

### Catalog Audit Gate

```powershell
python .agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py --include-green --show-skipped --defer deepinfra
```

Proves local AI SDK model-id unions are represented in mapped Siumai model sources, except documented deferrals.

### Focused Rust Gates

Run focused tests for edited model/catalog modules.

### Formatting Gate

```powershell
cargo fmt --check --package <touched-package>
```

Use a scoped command set if workspace formatting is too broad for the current change.

### Review Gate

Review should verify no old aliases were removed and no model list is duplicated in a new location.

## Evidence Anchors

- `.agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py`
- `docs/workstreams/provider-model-catalog-ai-sdk-refresh/TODO.md`
- Provider model constant files touched by PMCA-020.

## Fresh Evidence

| Date | Scope | Command | Result | Proves |
| --- | --- | --- | --- | --- |
| 2026-05-25 | Catalog audit | `python .agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py --include-green --show-skipped --defer deepinfra` | Pass: no red drift; DeepInfra deferred 51 of 91 upstream ids; green targets include Alibaba, Anthropic, Cohere, Google, Google Vertex, Mistral, xAI, and existing supported providers. | Low-risk provider model catalogs cover the local `repo-ref/ai` model-id unions; only documented DeepInfra policy work remains. |
| 2026-05-25 | Audit script syntax | `python -m py_compile .agents/skills/siumai-ai-sdk-maintenance/scripts/audit_model_catalogs.py` | Pass. | The repository-local maintenance script is syntactically valid Python. |
| 2026-05-25 | OpenAI-compatible model query path | `cargo nextest run -p siumai-provider-openai-compatible --features openai-standard --no-fail-fast providers::openai_compatible::providers::models::tests` | Pass: 14 passed, 227 skipped. | `get_models_for_provider` exposes refreshed Alibaba/Qwen, Mistral, xAI, and other OpenAI-compatible model sets; xAI `CHAT` and `popular` representative constants remain on the previous stable ids. |
| 2026-05-25 | Anthropic model constants | `cargo nextest run -p siumai-provider-anthropic --features anthropic --no-fail-fast providers::anthropic::model_constants::tests` | Pass: 2 passed, 98 skipped. | Refreshed Claude ids are included in chat model sets while Anthropic `popular` representative constants remain unchanged. |
| 2026-05-25 | Gemini default model list | `cargo nextest run -p siumai-provider-gemini --features google --no-fail-fast providers::gemini::models::tests` | Pass: 1 passed, 100 skipped. | Gemini default model list continues to cover current audited Google package ids. |
| 2026-05-25 | Cohere model constants | `cargo nextest run -p siumai-provider-cohere --features cohere --no-fail-fast providers::cohere::models::tests` | Pass: 3 passed, 26 skipped. | Cohere curated chat and embedding sets include the audited package ids. |
| 2026-05-25 | Google Vertex model constants | `cargo nextest run -p siumai-provider-google-vertex --features google-vertex --no-fail-fast providers::vertex::models::tests` | Pass: 4 passed, 106 skipped. | Vertex curated chat set includes refreshed Gemini ids and default model aggregation remains duplicate-free. |
| 2026-05-25 | xAI model constants | `cargo nextest run -p siumai-provider-xai --features xai --no-fail-fast providers::xai::models::tests` | Pass: 4 passed, 63 skipped. | Native xAI model sets, popular recommendations, and capability groups include refreshed Grok ids while keeping legacy ids. |
| 2026-05-25 | Registry catalog | `cargo nextest run -p siumai-registry --features all-providers --no-fail-fast provider_catalog` | Pass: 24 passed, 528 skipped. | Built-in provider catalog visibility includes refreshed Anthropic, Cohere, Vertex/Gemini, Mistral, xAI, and related provider entries. |
| 2026-05-25 | Public facade imports | `cargo nextest run -p siumai --features "openai google google-vertex cohere xai" --no-fail-fast public_surface_` | Pass: 32 passed, 800 skipped. | Refreshed ids are importable from public `siumai::provider_ext::*` paths, including OpenAI-compatible Alibaba/Qwen/Mistral, Gemini/Google, Vertex, Cohere, and xAI surfaces. |
| 2026-05-25 | Formatting | `cargo fmt --check --package siumai-provider-anthropic --package siumai-provider-cohere --package siumai-provider-gemini --package siumai-provider-google-vertex --package siumai-provider-openai-compatible --package siumai-provider-xai --package siumai-registry --package siumai` | Pass after scoped `cargo fmt` on the same package set. | All touched Rust packages are rustfmt-clean. |
| 2026-05-25 | Diff hygiene | `git diff --check -- .agents docs\workstreams\provider-model-catalog-ai-sdk-refresh siumai-provider-anthropic siumai-provider-cohere siumai-provider-gemini siumai-provider-google-vertex siumai-provider-openai-compatible siumai-provider-xai siumai-registry siumai` | Pass; Git reported only line-ending warnings. | No whitespace errors in the touched paths. |

## Notes

DeepInfra remains intentionally deferred. The audit script reports the exact deferred ids, and this lane should not silently expand that curated subset without a separate policy decision.

The first public-surface run used the same command and timed out after 304 seconds without failure output. It was rerun with a longer timeout and passed.
