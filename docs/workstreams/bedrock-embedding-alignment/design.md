# Amazon Bedrock Embedding Alignment - Design

Last updated: 2026-04-15

## Problem

Compared with `repo-ref/ai/packages/amazon-bedrock/src/index.ts` and
`bedrock-embedding-model.ts`, Siumai's Bedrock surface was only half-aligned:

- the upstream package exports `AmazonBedrockEmbeddingModelOptions`
- the upstream provider runtime exposes a real embedding lane
- the upstream provider supports Titan, Nova, and Cohere embedding request/response shapes

The Rust side was still missing the whole embedding lane:

- no public `AmazonBedrockEmbeddingModelOptions`
- no provider-owned `EmbeddingCapability` on `BedrockClient`
- no `registry.embedding_model("bedrock:...")` factory path
- public and lower-contract tests intentionally locked Bedrock embedding as unsupported

That left Bedrock in an awkward state where the package boundary already advertised embedding in the
reference repo, but the Rust wrapper still behaved as a chat+rerank-only slice at the time of this
audit.

## Goals

- Audit the Bedrock embedding lane against the upstream AI SDK package and runtime.
- Add the missing public typed embedding options on the provider-owned and facade surfaces.
- Implement a real provider-owned Bedrock embedding lane instead of only exporting a placeholder
  type.
- Make builder/config-first/registry/public paths converge on the same final `/model/{id}/invoke`
  request shape.
- Keep the implementation aligned with the upstream request/response family split:
  Titan, Nova, Cohere v3, and Cohere v4.

## Non-goals

- Do not fold Bedrock image parity into this embedding workstream; image now has its own follow-up
  under `docs/workstreams/bedrock-image-alignment/`.
- Do not overfit the Rust API to a single Bedrock model family.
- Do not widen Bedrock into unrelated provider-owned extras just because the generic runtime can
  execute JSON POST requests.

## Chosen design

### 1. Add the missing typed embedding option surface

Rust now exposes the missing public embedding option alias:

- `AmazonBedrockEmbeddingModelOptions`

The provider-owned typed surface also models the important Bedrock-specific option subsets that are
encoded as TypeScript unions upstream:

- `BedrockEmbeddingInputType`
- `BedrockEmbeddingPurpose`
- `BedrockEmbeddingTruncate`
- `BedrockEmbeddingOptions`

This keeps the Bedrock package surface easier to compare against the upstream AI SDK export
boundary while still giving Rust callers typed constructors instead of raw JSON objects.

### 2. Implement a real provider-owned embedding standard

Instead of bolting embedding onto Bedrock with ad hoc request code in the client, the new runtime
is implemented as a first-class Bedrock embedding standard:

- shared Bedrock JSON header helper
- Bedrock embedding spec for `/model/{id}/invoke`
- request transformer that branches by model family
- response transformer that accepts Titan, Nova, Cohere v3, and Cohere v4 response shapes

That keeps Bedrock embedding aligned with the existing chat/rerank provider architecture and avoids
creating another one-off execution path.

### 3. Keep the upstream family split intact

The request body is shaped using the same model-family split as the AI SDK reference:

- Titan:
  - `inputText`
  - `dimensions`
  - `normalize`
- Nova:
  - `taskType: "SINGLE_EMBEDDING"`
  - `singleEmbeddingParams.embeddingPurpose`
  - `singleEmbeddingParams.embeddingDimension`
  - `singleEmbeddingParams.text.truncationMode`
  - `singleEmbeddingParams.text.value`
- Cohere:
  - `input_type`
  - `texts`
  - `truncate`
  - `output_dimension`

The Rust implementation also adds safe validation for the audited dimension subsets before
transport, because the upstream zod schema already encodes those same limits.

### 4. Promote embedding to a first-class Bedrock capability

`BedrockClient`, the native provider metadata, and the registry factory now all advertise embedding
support. This removes the earlier fake "unsupported before transport" boundary and makes the
following construction paths converge:

- provider-owned builder/config-first clients
- `Provider::bedrock()` / `Siumai::builder().bedrock()`
- `registry.embedding_model("bedrock:...")`

### 5. Update the historical unsupported docs/tests

The previous docs/tests that intentionally locked Bedrock embedding as unsupported are now updated
to request-shape parity coverage instead of fail-fast assertions. That keeps the regression suite in
sync with the new provider reality.

## Validation

This workstream is locked by:

- typed option serialization/deserialization tests in
  `siumai-provider-amazon-bedrock/src/provider_options/bedrock.rs`
- request/response transformer tests in
  `siumai-provider-amazon-bedrock/src/standards/bedrock/embedding.rs`
- public surface compile guards in `siumai/tests/public_surface_imports_test.rs`
- top-level public-path parity tests in `siumai/tests/provider_public_path_parity_test.rs`
- lower factory/registry contract tests in
  `siumai-registry/src/registry/factories/contract_tests.rs`

## Remaining follow-up

- Re-audit whether the upstream package eventually exports a public Bedrock embedding model id type
  that deserves a direct Rust facade mirror.
- Re-check whether additional Bedrock embedding model families appear upstream and should be added
  to the model-family routing heuristics.
- Decide later whether Bedrock request-ext helpers should be expanded further beyond the newly added
  embedding option helper lane.
