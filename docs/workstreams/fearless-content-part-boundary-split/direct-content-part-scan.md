# Direct ContentPart Construction Scan

Last updated: 2026-05-16

## Scan Command

```text
rg -l "ContentPart::|provider_metadata:|provider_options:" \
  siumai-core/src siumai-bridge/src \
  siumai-protocol-openai/src siumai-protocol-anthropic/src siumai-protocol-gemini/src \
  siumai-provider-amazon-bedrock/src siumai-provider-anthropic/src \
  siumai-provider-google-vertex/src siumai-provider-minimaxi/src \
  siumai-provider-openai/src siumai-provider-gemini/src siumai/src -g "*.rs"
```

## 2026-05-16 Refresh

The refreshed scan found:

- 121 source files already classified by
  `docs/workstreams/fearless-spec-core-boundary-convergence/content-part-construction-audit.md`.
- 48 additional low-priority or false-positive paths that were previously hidden behind broad
  guard buckets such as `/provider_ext/`, `/mod.rs`, `/builder.rs`, `/config.rs`, and `/tests.rs`.
- 0 unclassified high-value production paths.

The facade boundary guard now requires every matching source path to appear in either the original
spec/core audit or this refreshed scan document. There is no longer an automatic path-suffix
allowlist for future hits.

## Explicit Low-Priority Or False-Positive Paths

| Path | Classification |
| --- | --- |
| `siumai-bridge/src/request/tests.rs` | inline bridge request tests |
| `siumai-bridge/src/response/tests.rs` | inline bridge response tests |
| `siumai-bridge/src/stream/tests.rs` | inline bridge stream tests |
| `siumai-core/src/custom_provider/mod.rs` | custom-provider module shell and docs |
| `siumai-core/src/streaming/builder.rs` | already guarded core stream helper path |
| `siumai-core/src/utils/mod.rs` | utility module shell |
| `siumai-protocol-anthropic/src/standards/anthropic/streaming/tests.rs` | inline Anthropic streaming tests |
| `siumai-protocol-anthropic/src/standards/anthropic/utils/mod.rs` | Anthropic utility module shell |
| `siumai-protocol-gemini/src/standards/gemini/streaming/tests.rs` | inline Gemini streaming tests |
| `siumai-protocol-openai/src/standards/openai/responses_sse/tests.rs` | inline OpenAI Responses SSE tests |
| `siumai-provider-amazon-bedrock/src/provider_options/bedrock.rs` | request-side provider option carrier |
| `siumai-provider-anthropic/src/provider_metadata/mod.rs` | provider metadata re-export shell |
| `siumai-provider-anthropic/src/providers/anthropic/builder.rs` | provider builder request defaults |
| `siumai-provider-anthropic/src/providers/anthropic/config.rs` | provider config request defaults |
| `siumai-provider-anthropic/src/providers/anthropic/mod.rs` | provider module shell |
| `siumai-provider-gemini/src/provider_options/gemini/mod.rs` | request-side provider option module shell |
| `siumai-provider-gemini/src/providers/gemini/mod.rs` | provider module shell |
| `siumai-provider-google-vertex/src/provider_options/vertex/imagen.rs` | request-side Vertex Imagen provider options |
| `siumai-provider-google-vertex/src/providers/anthropic_vertex/builder.rs` | provider builder request defaults |
| `siumai-provider-google-vertex/src/providers/anthropic_vertex/mod.rs` | provider module shell |
| `siumai-provider-google-vertex/src/providers/vertex/mod.rs` | provider module shell |
| `siumai-provider-minimaxi/src/providers/minimaxi/builder.rs` | provider builder request defaults |
| `siumai-provider-minimaxi/src/providers/minimaxi/config.rs` | provider config request defaults |
| `siumai-provider-minimaxi/src/providers/minimaxi/mod.rs` | provider module shell |
| `siumai-provider-minimaxi/src/providers/minimaxi/tests.rs` | inline MiniMaxi tests |
| `siumai-provider-openai/src/provider_metadata/mod.rs` | provider metadata re-export shell |
| `siumai-provider-openai/src/providers/openai/builder.rs` | provider builder request defaults |
| `siumai-provider-openai/src/providers/openai/config.rs` | provider config request defaults |
| `siumai-provider-openai/src/providers/openai/mod.rs` | provider module shell |
| `siumai/src/provider_ext/anthropic.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/azure.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/bedrock.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/cohere.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/deepseek.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/fireworks.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/gemini.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/google_vertex.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/groq.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/minimaxi.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/mistral.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/moonshotai.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/ollama.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/openai.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/openai_compatible.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/openrouter.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/perplexity.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/togetherai.rs` | facade provider extension re-export shell |
| `siumai/src/provider_ext/xai.rs` | facade provider extension re-export shell |
