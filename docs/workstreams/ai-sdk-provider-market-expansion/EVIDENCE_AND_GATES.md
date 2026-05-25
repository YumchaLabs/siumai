# AI SDK Provider Market Expansion — Evidence And Gates

Status: Closed
Last updated: 2026-05-26

## Market Evidence

Source: npm downloads API, `last-month`.

Window returned by npm: `2026-04-25` through `2026-05-24`.

Representative query pattern:

```text
https://api.npmjs.org/downloads/point/last-month/%40ai-sdk%2Fopenai
```

Current ranking evidence:

| Package | Downloads |
| --- | ---: |
| `@ai-sdk/gateway` | 45,995,757 |
| `@ai-sdk/openai` | 26,415,755 |
| `@ai-sdk/anthropic` | 23,989,010 |
| `@ai-sdk/google` | 17,601,231 |
| `@ai-sdk/openai-compatible` | 11,051,338 |
| `@ai-sdk/google-vertex` | 6,609,651 |
| `@ai-sdk/amazon-bedrock` | 6,285,133 |
| `@ai-sdk/xai` | 5,841,870 |
| `@ai-sdk/azure` | 5,364,901 |
| `@ai-sdk/groq` | 4,401,992 |
| `@ai-sdk/mistral` | 4,142,636 |
| `@ai-sdk/deepseek` | 3,458,059 |
| `@ai-sdk/cerebras` | 2,859,534 |
| `@ai-sdk/perplexity` | 2,712,137 |
| `@ai-sdk/togetherai` | 2,090,503 |
| `@ai-sdk/cohere` | 1,742,523 |
| `@ai-sdk/fireworks` | 981,722 |
| `@ai-sdk/deepgram` | 605,045 |
| `@ai-sdk/elevenlabs` | 533,132 |
| `@ai-sdk/deepinfra` | 433,919 |
| `@ai-sdk/replicate` | 257,948 |
| `@ai-sdk/fal` | 192,339 |

Interpretation caveats:

- Downloads are not market share. They include CI, lockfile churn, and transitive dependencies.
- `@ai-sdk/gateway` is partly inflated because the `ai` package depends on and re-exports it.
- The transitive gateway signal still matters because the AI SDK runtime defaults to gateway when no custom
  global provider is configured.

## Baseline Gates

Use these before broad provider changes:

```powershell
python .agents/skills/siumai-ai-sdk-maintenance/scripts/resolve_ai_sdk_repo.py
scripts\audit-model-catalogs.bat
cargo fmt --check
git diff --check
```

## Focused Gates By Task

Gateway:

```powershell
cargo nextest run -p siumai-provider-gateway --features gateway gateway --no-fail-fast
cargo nextest run -p siumai-registry --features gateway gateway --no-fail-fast
cargo nextest run -p siumai --features gateway --test gateway_provider_surface_test --no-fail-fast
```

Cerebras:

```powershell
cargo nextest run -p siumai-provider-openai-compatible cerebras --features openai-standard --no-fail-fast
cargo nextest run -p siumai-registry cerebras --features openai --no-fail-fast
cargo nextest run -p siumai cerebras --features openai --no-fail-fast
```

Mistral:

```powershell
cargo nextest run -p siumai-protocol-openai --features openai-standard openai_compatible_mistral --no-fail-fast
cargo nextest run -p siumai-provider-openai-compatible mistral --features openai-standard --no-fail-fast
cargo nextest run -p siumai-registry --features openai mistral --no-fail-fast
cargo nextest run -p siumai --features openai --test provider_public_path_parity_test mistral_ --no-fail-fast
cargo nextest run -p siumai --features openai --test public_surface_imports_test public_surface_mistral_fireworks_perplexity_provider_ext_compile --no-fail-fast
cargo nextest run -p siumai --features openai --test mistral_openai_compat_url_alignment_test --no-fail-fast
```

Enterprise providers:

```powershell
cargo nextest run -p siumai-provider-azure --no-fail-fast
cargo nextest run -p siumai-provider-amazon-bedrock --no-fail-fast
cargo nextest run -p siumai-provider-google-vertex --no-fail-fast
```

Audio/media decision:

```powershell
python .agents/skills/siumai-ai-sdk-maintenance/scripts/resolve_ai_sdk_repo.py
```

## Evidence Log

| Date | Task | Evidence | Result |
| --- | --- | --- | --- |
| 2026-05-25 | PMX-010 | Workstream opened with npm download evidence and local AI SDK package inventory. | Done |
| 2026-05-25 | PMX-020 | `GATEWAY_INVENTORY.md` created from `repo-ref/ai/packages/gateway`. | Done |
| 2026-05-25 | PMX-030 | `python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py` | Pass: resolved local AI SDK reference at `repo-ref/ai`. |
| 2026-05-25 | PMX-030 | `cargo nextest run -p siumai-provider-gateway --features gateway gateway --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-gateway` because `F:` had 0 bytes free. | Pass: 5 tests run, 5 passed. Covers Gateway V4 language non-stream request URL/body/headers, language SSE stream request/part parsing, chat compatibility bridge for non-stream and stream, and embedding request/response/provider options. |
| 2026-05-25 | PMX-030 | `cargo nextest run -p siumai-registry --features gateway gateway --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-gateway`. | Pass: 5 tests run, 5 passed, 123 skipped. Covers `SiumaiBuilder::gateway()`, native registry registration, provider catalog metadata, factory context precedence/request headers, language/embedding family construction, and unsupported follow-on families before transport use. Narrow gateway-only test build emits existing unused helper warnings in `contract_tests.rs`. |
| 2026-05-25 | PMX-030 | `cargo nextest run -p siumai --features gateway --test gateway_provider_surface_test --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-gateway`. | Pass: 2 tests run, 2 passed. Covers `provider_ext::gateway`, `providers::gateway`, `Provider::gateway()`, `create_gateway()`, `GatewayConfig`, and typed `GatewayOptions` request extension exports. |
| 2026-05-25 | PMX-030 | `cargo fmt --check --package siumai-provider-gateway --package siumai-registry --package siumai --package siumai-spec` | Pass. |
| 2026-05-25 | PMX-030 | `cargo check -p siumai-provider-gateway --tests --features gateway`; `cargo check -p siumai-registry --tests --features gateway`; `cargo check -p siumai --tests --features gateway` | Pass. Initial broad `cargo nextest run -p siumai --features gateway gateway_provider_surface_test` on the default `F:\...\target` failed before tests due `F:` being full, corrupt MSVC PDB output, and rustc `no space on device`; superseded by the targeted D-drive nextest pass above. |
| 2026-05-25 | PMX-030 | `git diff --check` | Pass: no whitespace errors. Git emitted LF-to-CRLF working-copy warnings for existing dirty files. |
| 2026-05-25 | PMX-040 | `repo-ref/ai/packages/cerebras/src/{cerebras-provider.ts,cerebras-chat-language-model.ts,cerebras-chat-options.ts}` compared against Siumai OpenAI-compatible provider patterns. | Done: implementation boundary kept to OpenAI-compatible chat/language model; embedding/image and other families remain unsupported. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai-provider-openai-compatible cerebras --features openai-standard --no-fail-fast` | Pass: 8 tests run, 8 passed, 241 skipped. Covers builtin preset, model catalog, provider settings, structured output default, `reasoning_content` -> `reasoning` request-body rewrite, Cerebras GLM structured-output `tool_calls` finish normalization for non-stream and stream paths, and the no-tool-call-parts text edge case from the AI SDK predicate. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai-provider-openai-compatible promoted_vendor_clients_only_expose_explicit_completion_capability --features openai-standard --no-fail-fast` | Pass: 1 test run, 1 passed, 248 skipped. Covers Cerebras staying out of explicit completion capability exposure alongside other chat-only promoted compat presets. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai-registry cerebras --features openai --no-fail-fast` | Pass: 3 tests run, 3 passed, 242 skipped. Covers `SiumaiBuilder::cerebras()`, provider catalog id/type/default URL/models, and factory rejection of non-text family paths before transport use. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai cerebras --features openai --no-fail-fast` | Pass: 2 tests run, 2 passed, 542 skipped. Covers `siumai::provider_ext::cerebras`, `siumai::providers::cerebras`, `Provider::cerebras()`, settings, model constants, and first-class metadata classification. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai-spec provider_type --no-fail-fast` | Pass: 17 tests run, 17 passed, 250 skipped. Covers `ProviderType::Cerebras` mapping alongside existing provider identity checks. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai --test cerebras_provider_surface_test --features openai --no-fail-fast` | Pass: 2 tests run, 2 passed, 0 skipped. Covers the dedicated facade public-path test target after the broader `siumai cerebras` filter initially hit a compile timeout. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai --test siumai_unified_interface_test -E 'test(test_provider_type_consistency)' --no-default-features --features openai,azure,anthropic,google,xai,ollama,groq,deepseek --no-fail-fast` | Pass: 1 test run, 1 passed, 9 skipped. Covers full-feature facade `ProviderType` consistency including `ProviderType::Cerebras`. |
| 2026-05-25 | PMX-040 | `python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py` | Pass: resolved local AI SDK reference at `repo-ref/ai`. |
| 2026-05-25 | PMX-040 | `scripts\audit-model-catalogs.bat` | Pass: no red model catalog drift; Cerebras is green with 6 upstream ids covered; DeepInfra remains explicitly deferred. |
| 2026-05-25 | PMX-040 | `cargo fmt --check -p siumai -p siumai-spec -p siumai-provider-openai-compatible -p siumai-registry -p siumai-protocol-openai` | Pass. |
| 2026-05-25 | PMX-040 | `git diff --check` | Pass. |
| 2026-05-25 | PMX-040 | `cargo nextest run -p siumai --no-default-features --features openai,azure,anthropic,google,xai,ollama,groq,deepseek -E 'test(test_provider_type_consistency)' --no-fail-fast` | Earlier attempt timed out after 304 seconds during compilation due command shape and broad feature compile cost; superseded by the explicit `--test siumai_unified_interface_test -E 'test(test_provider_type_consistency)'` pass above. |
| 2026-05-25 | PMX-050 | `repo-ref/ai/packages/mistral/src/{mistral-provider.ts,mistral-chat-language-model.ts,mistral-chat-language-model-options.ts,mistral-embedding-model.ts,mistral-embedding-options.ts,convert-to-mistral-chat-messages.ts,mistral-prepare-tools.ts,map-mistral-finish-reason.ts}` compared against Siumai Mistral preset/facade sources. | Done: Mistral remains an OpenAI-compatible provider. Concrete bounded gaps were limited to request-body normalization for unsupported `topK`, `stopSequences` preservation as `stop`, and the current `reasoningEffort` model support list. See `MISTRAL_AUDIT.md`. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai-protocol-openai --features openai-standard openai_compatible_mistral --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-mistral`. | Pass: 7 tests run, 7 passed, 345 skipped. Covers Mistral request body normalization for unsupported common settings, `stopSequences` -> `stop`, JSON schema/object behavior, provider-option cleanup, reasoning-effort support list, and tool-choice mapping. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai-provider-openai-compatible mistral --features openai-standard --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-mistral`. | Pass: 6 tests run, 6 passed, 243 skipped. Covers Mistral model constants, typed options serialization/aliases, request extension merge, provider settings, and runtime `model_length` finish normalization. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai-registry --features openai mistral --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-mistral`. | Pass: 4 tests run, 4 passed, 241 skipped. Covers provider catalog mapping, native completion-family rejection, explicit completion capability absence, and Mistral embedding registry override behavior. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai --features openai --test public_surface_imports_test public_surface_mistral_fireworks_perplexity_provider_ext_compile --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-mistral`. | Pass: 1 test run, 1 passed, 22 skipped. Covers facade import paths for Mistral builder helpers, model constants, and typed options. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai --features openai --test provider_public_path_parity_test mistral_ --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-mistral`. | Pass: 9 tests run, 9 passed, 141 skipped. Covers Mistral builder/config/registry chat, chat stream, embedding, options, package settings, and unsupported completion-family behavior. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai --features openai --test mistral_openai_compat_url_alignment_test --no-fail-fast` with `CARGO_TARGET_DIR=D:\siumai-target-mistral`. | Pass: 2 tests run, 2 passed. Covers Mistral chat and embedding endpoint URLs. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai-registry mistral --no-fail-fast` | Earlier attempt ran without the `openai` feature and discovered 0 tests; superseded by the feature-correct registry pass above. |
| 2026-05-25 | PMX-050 | `cargo nextest run -p siumai mistral --no-fail-fast` | Earlier broad facade filter timed out after 304 seconds during compilation; superseded by explicit facade test-target passes above. |
| 2026-05-25 | PMX-050 | `cargo fmt --check -p siumai-protocol-openai`; `python -m json.tool docs\workstreams\ai-sdk-provider-market-expansion\WORKSTREAM.json`; `git diff --check` | Pass. `git diff --check` emitted only LF-to-CRLF working-copy warnings for touched files. |
| 2026-05-25 | PMX-060 | `repo-ref/ai/packages/azure/src/{index.ts,azure-openai-provider.ts,azure-openai-tools.ts,azure-openai-provider-metadata.ts}` compared against Siumai Azure provider, facade, registry, and fixture tests. | Done: Azure already covers Responses default, Chat Completions, completion, embedding, image, speech, transcription, deployment/v1 URL modes, typed options, hosted tools, and metadata through no-network tests. |
| 2026-05-25 | PMX-060 | `repo-ref/ai/packages/amazon-bedrock/src/{index.ts,amazon-bedrock-provider.ts,anthropic/index.ts,mantle/index.ts}` compared against Siumai Bedrock provider, facade, registry, and fixture tests. | Done: core Bedrock chat/embedding/image/rerank surface is covered. PMX-060 added the bounded `amazon_bedrock()` alias. Bedrock Anthropic and Bedrock Mantle are follow-ons because they are distinct sub-provider implementations. |
| 2026-05-25 | PMX-060 | `repo-ref/ai/packages/google-vertex/src/{index.ts,google-vertex-provider.ts,google-vertex-provider-base.ts,edge/index.ts,anthropic/index.ts,maas/index.ts,xai/index.ts}` compared against Siumai Vertex, Anthropic-on-Vertex, Vertex MaaS, and Vertex xAI surfaces. | Done: Google Vertex main and subpath surfaces are represented by Rust provider settings, token providers, facade modules, registry aliases, model constants, and no-network request/fixture tests. |
| 2026-05-25 | PMX-060 | `cargo nextest run -p siumai --features bedrock --test public_surface_imports_test public_surface_bedrock_provider_ext_compiles --no-fail-fast` | Pass: 1 test run, 1 passed, 23 skipped. Covers Bedrock facade imports including `amazon_bedrock()`, `create_amazon_bedrock()`, typed settings/options/metadata, compat `Provider::amazon_bedrock()`, and `SiumaiBuilder::amazon_bedrock()`. |
| 2026-05-25 | PMX-060 | `cargo nextest run -p siumai-registry --features bedrock amazon_bedrock --no-fail-fast` | Pass: 1 test run, 1 passed, 144 skipped. Covers `SiumaiBuilder::amazon_bedrock()` resolving to the canonical Bedrock provider id. Narrow build emitted existing unused helper warnings in `contract_tests.rs`. |
| 2026-05-25 | PMX-060 | `cargo fmt --check -p siumai -p siumai-registry`; `git diff --check` | Pass. `git diff --check` emitted only LF-to-CRLF working-copy warnings for touched files. |
| 2026-05-26 | PMX-070 | npm downloads API `last-month` for `@ai-sdk/deepgram`, `@ai-sdk/elevenlabs`, `@ai-sdk/fal`, and `@ai-sdk/replicate`. | Pass: npm returned the same frozen window `2026-04-25` through `2026-05-24`: Deepgram 605,045; ElevenLabs 533,132; Replicate 257,948; Fal 192,339. |
| 2026-05-26 | PMX-070 | `repo-ref/ai/packages/{deepgram,elevenlabs,fal,replicate}/src` package surfaces compared against Siumai media families and registry handles. | Done: Deepgram and ElevenLabs are narrow speech/transcription candidates; Replicate and Fal require dedicated media lanes because of prediction/queue polling, timeout policy, dynamic model inputs, and broad model-specific request options. |
| 2026-05-26 | PMX-070 | `docs/workstreams/ai-sdk-provider-market-expansion/AUDIO_MEDIA_DECISION.md` | Done: decision note ranks provider priority, defers broad media implementation, and splits follow-ons for Deepgram, ElevenLabs, shared media polling policy, Replicate, and Fal. |
| 2026-05-26 | PMX-070 | `python .agents\skills\siumai-ai-sdk-maintenance\scripts\resolve_ai_sdk_repo.py`; `python -m json.tool docs\workstreams\ai-sdk-provider-market-expansion\WORKSTREAM.json`; `git diff --check` | Pass: AI SDK reference resolved at `repo-ref/ai`; WORKSTREAM.json parsed successfully; `git diff --check` found no whitespace errors and emitted only LF-to-CRLF working-copy warnings for touched docs. |
| 2026-05-26 | PMX-080 | Workstream closeout review of `DESIGN.md`, `TODO.md`, `MILESTONES.md`, `EVIDENCE_AND_GATES.md`, `WORKSTREAM.json`, `HANDOFF.md`, and `GATEWAY_INVENTORY.md`; WORKSTREAM JSON parse check; stale status marker grep; `git diff --check`. | Pass: all task items are complete or split; no stale Draft, open PMX task, or ready-to-close markers remain; WORKSTREAM.json parsed successfully; `git diff --check` found no whitespace errors and emitted only LF-to-CRLF working-copy warnings for touched docs. |
