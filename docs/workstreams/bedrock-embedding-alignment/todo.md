# Amazon Bedrock Embedding Alignment - TODO

Last updated: 2026-04-15

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[-]` intentionally deferred

## Track A - Public type parity

- [x] Audit `repo-ref/ai/packages/amazon-bedrock/src/index.ts`.
- [x] Add `AmazonBedrockEmbeddingModelOptions` on the provider-owned/public Rust surface.
- [x] Add typed Rust enums/helpers for the important embedding-only Bedrock option subsets.
- [x] Re-export the new embedding option surface from `provider_ext::bedrock::{options::*, *}`.

## Track B - Provider-owned embedding runtime

- [x] Add a first-class Bedrock embedding standard instead of a one-off client-local request path.
- [x] Route embedding requests to `/model/{id}/invoke`.
- [x] Implement Titan embedding request shaping.
- [x] Implement Nova embedding request shaping.
- [x] Implement Cohere embedding request shaping.
- [x] Parse Titan, Nova, Cohere v3, and Cohere v4 embedding responses.
- [x] Validate the audited Bedrock dimension subsets before transport.

## Track C - Client / registry parity

- [x] Make `BedrockClient` expose `EmbeddingCapability` / `EmbeddingExtensions`.
- [x] Advertise Bedrock embedding in native provider metadata and factory capabilities.
- [x] Add `embedding_model_with_ctx(...)` on the Bedrock factory.
- [x] Replace the old fail-fast Bedrock embedding contract tests with request-shape parity tests.
- [x] Replace the old fail-fast public-path Bedrock embedding tests with request-shape parity tests.

## Track D - Docs and changelog

- [x] Create a dedicated `docs/workstreams/bedrock-embedding-alignment/` folder.
- [x] Update `CHANGELOG.md` `Unreleased` with the Bedrock embedding/runtime parity changes.
- [x] Update the old fearless-refactor notes that still described Bedrock embedding as unsupported.

## Track E - Intentional deferrals

- [x] Keep Bedrock image parity out of this embedding workstream; that follow-up now lives in
  `docs/workstreams/bedrock-image-alignment/`.
- [-] Do not mirror TypeScript-only `createAmazonBedrock`, `bedrock`, or provider-settings exports
  as direct Rust package facades in this workstream.
