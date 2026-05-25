# PMX-060 Enterprise Provider Polish Audit

Status: Done
Date: 2026-05-25

## Scope

This audit compares the local AI SDK reference packages for Azure, Amazon Bedrock, and Google
Vertex against Siumai's current provider crates, registry integration, facade exports, and
no-network test coverage.

Upstream authority:

- `repo-ref/ai/packages/azure/src/index.ts`
- `repo-ref/ai/packages/azure/src/azure-openai-provider.ts`
- `repo-ref/ai/packages/azure/src/azure-openai-tools.ts`
- `repo-ref/ai/packages/amazon-bedrock/src/index.ts`
- `repo-ref/ai/packages/amazon-bedrock/src/amazon-bedrock-provider.ts`
- `repo-ref/ai/packages/amazon-bedrock/src/anthropic/index.ts`
- `repo-ref/ai/packages/amazon-bedrock/src/mantle/index.ts`
- `repo-ref/ai/packages/google-vertex/src/index.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-provider.ts`
- `repo-ref/ai/packages/google-vertex/src/google-vertex-provider-base.ts`
- `repo-ref/ai/packages/google-vertex/src/edge/index.ts`
- `repo-ref/ai/packages/google-vertex/src/{anthropic,maas,xai}/index.ts`

Siumai authority:

- `siumai-provider-azure/src/providers/azure_openai/*`
- `siumai-provider-amazon-bedrock/src/providers/bedrock/*`
- `siumai-provider-google-vertex/src/providers/{vertex,anthropic_vertex}/*`
- `siumai-provider-openai-compatible/src/providers/openai_compatible/*vertex*`
- `siumai-registry/src/provider_builders.rs`
- `siumai-registry/src/provider_catalog.rs`
- `siumai-registry/src/registry/factories/{azure,bedrock,google_vertex,anthropic_vertex,vertex_maas,google_vertex_xai}.rs`
- `siumai/src/provider_ext/{azure,bedrock,google_vertex,anthropic_vertex,vertex_maas,google_vertex_xai}.rs`
- `siumai/tests/provider_public_path_parity/*{azure,bedrock,vertex}*`
- `siumai/tests/public_surface_imports_test.rs`

## Decision

Azure, Bedrock, and Google Vertex already have meaningful no-network coverage and provider-owned
typed surfaces in Siumai. PMX-060 therefore stays a polish task, not a rewrite or provider
onboarding task.

The only bounded API polish fix from this audit is an Amazon Bedrock naming alias:
`amazon_bedrock()` now mirrors AI SDK's canonical `amazonBedrock` export while preserving the older
Rust `bedrock()` spelling as a convenience alias.

Do not fold Bedrock Anthropic or Bedrock Mantle implementation into this workstream. Both are
sub-provider surfaces with separate transport semantics and should be split into dedicated
follow-on lanes if prioritized.

## Findings

| Provider | Area | Upstream AI SDK behavior | Siumai result | Status |
| --- | --- | --- | --- | --- |
| Azure | Package factory | `azure`, `createAzure`, `resourceName`, `baseURL`, `apiKey`, headers/fetch, `apiVersion`, and `useDeploymentBasedUrls`. | `provider_ext::azure::{azure, create_azure}`, `AzureOpenAIProviderSettings`, `AzureUrlConfig`, deployment and v1 URL modes, custom transport/header support. | Green |
| Azure | Families | Default callable/language route uses Responses; explicit `chat`, `responses`, `completion`, `embedding`, `image`, `transcription`, and `speech`. | Facade, config, provider, and registry public-path tests cover responses/chat, completion, embedding, image, speech, and transcription URLs and request equivalence. | Green |
| Azure | Tools and metadata | Re-exports OpenAI tools under Azure and wraps Responses metadata under the `azure` key. | Siumai exposes Azure typed provider options/metadata and fixture tests cover web search preview, code interpreter, image generation, file search, reasoning encrypted content, and raw provider metadata. | Green |
| Bedrock | Package factory | Canonical `amazonBedrock`, deprecated `bedrock`, and `createAmazonBedrock`; region/API-key/SigV4/baseURL/headers/fetch settings. | PMX-060 added `provider_ext::bedrock::amazon_bedrock()`, `Provider::amazon_bedrock()`, and `SiumaiBuilder::amazon_bedrock()` aliases. Existing settings support region, bearer API key, base URL, headers, and custom transport; first-class SigV4 credential-provider fields remain intentionally deferred. | Fixed with documented divergence |
| Bedrock | Families | Converse language, embedding, image, reranking, Anthropic tools, plus deprecated text-embedding aliases. | No-network public-path tests cover chat, stream, embedding, image, rerank, tools, file citations/cache points, and provider metadata. Public surface exports typed Bedrock options and Anthropic provider options when the Anthropic feature is enabled. | Green |
| Bedrock | Subpaths | `@ai-sdk/amazon-bedrock/anthropic` wraps Anthropic Messages over Bedrock InvokeModel; `@ai-sdk/amazon-bedrock/mantle` wraps the OpenAI-compatible Mantle endpoint. | Siumai does not expose Bedrock Anthropic or Mantle as first-class provider packages. These are broader than polish because they need distinct factories, auth/signing behavior, model catalogs, and no-network request/stream gates. | Follow-on |
| Google Vertex | Main package | `googleVertex`, `createGoogleVertex`, deprecated `vertex`/`createVertex`, API-key express mode, project/location enterprise mode, baseURL, headers/fetch, Node auth and edge credentials paths. | `provider_ext::google_vertex::{google_vertex, create_google_vertex, vertex, create_vertex}`, `GoogleVertexProviderSettings`, token-provider auth analogue, base URL helpers, express/enterprise URL tests, and GCP auth alignment tests. | Green |
| Google Vertex | Families | Gemini language, embedding, image, video, and Google-hosted tools. | Siumai has typed options/model constants for chat, embedding, Imagen image/edit, and video; fixture tests cover chat, embeddings, Imagen generation/edit, video helpers, provider tools, and metadata. | Green |
| Google Vertex | Subpaths | `anthropic`, `maas`, and `xai` package subpaths with edge variants. | Siumai exposes `anthropic_vertex`, `vertex_maas`, and `google_vertex_xai` provider extension modules, registry aliases, model constants, public-surface tests, and request parity tests. Edge-vs-Node split is intentionally collapsed into Rust auth/provider settings. | Green with Rust API divergence |

## Fixed Gap

- Added the Rust package-surface alias `amazon_bedrock()` alongside the existing `bedrock()` helper:
  - `siumai::provider_ext::bedrock::amazon_bedrock()`
  - `siumai::providers::bedrock::amazon_bedrock()`
  - `siumai::compat::Provider::amazon_bedrock()`
  - `SiumaiBuilder::amazon_bedrock()`

This closes a small naming mismatch with AI SDK's current canonical export without changing the
existing provider id, registry routing, or request behavior.

## Intentional Divergences

- Rust does not mirror JavaScript callable provider objects one-for-one. The stable Rust shape is
  builder/config/registry/facade based.
- Bedrock's first-class AWS SigV4 credential-provider abstraction is deferred. Current Siumai usage
  can use bearer API key auth, explicit headers, custom transport, or pre-signed/custom HTTP layers.
- Google Vertex's Node and edge package split is represented by Rust provider settings and token
  providers rather than separate public modules.
- Bedrock Anthropic and Bedrock Mantle are not PMX-060 polish fixes. They should be tracked as
  dedicated follow-ons if market evidence justifies the implementation cost.
- Live credential tests remain out of scope. PMX-060 relies on no-network public-surface, URL,
  request-body, stream, metadata, and provider catalog gates.

## Validation

See `EVIDENCE_AND_GATES.md` for fresh command output recorded after this audit.
