# Vertex MaaS Unified Provider Surface - Design

Last updated: 2026-04-05

## Problem

AI SDK 在 `repo-ref/ai/packages/google-vertex/src/maas/*` 里把 Vertex MaaS 建模成一个一等
provider surface：

- `chatModel()`
- `completionModel()`
- `embeddingModel()`
- `textEmbeddingModel()`

底层实现并不是一套新的协议，而是：

- 复用 `openai-compatible` runtime
- 基于 `project + location` 组装
  `https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi`
- 在 Node/Edge wrapper 里注入 Google Bearer token

Siumai 之前有三个关键缺口：

1. `vertex-maas` 不是一等 built-in provider id
2. 公共 builder / registry / provider catalog 没有 AI SDK 风格的统一 surface
3. shared OpenAI-compatible runtime 会在 `api_key` 为空时过早报错，导致已有
   `Authorization: Bearer ...` 或 token provider 的 Google auth 语义根本走不通

## Goals

- 让 `vertex-maas` 成为一等 built-in provider id。
- 提供 AI SDK 风格的统一 public surface：
  - `Provider::vertex_maas()`
  - `Siumai::builder().vertex_maas()`
  - registry `language_model/completion_model/embedding_model`
- 复用 shared OpenAI-compatible runtime，而不是新造一套 provider crate。
- 对齐 AI SDK 的 `project + location -> openapi base URL` 语义。
- 对齐 Google Bearer auth 语义，而不是继续依赖“伪 API key”兼容技巧。
- 让稳定层 `ProviderType`、provider catalog、validator、retry 也知道这个 provider。

## Non-goals

- 不新建 `siumai-provider-google-vertex-maas` crate。
- 不把 Vertex MaaS 伪装成支持 image / rerank / speech / transcription。
- 不因为 AI SDK `OpenAICompatibleProvider` 的泛型继承就机械宣称 `imageModel()` 已完成对齐。
  目前上游 README、examples、以及 MaaS 专门测试只实锤了 text/completion/embedding 主路径，
  没有把 image 当成正式 audited surface。
- 不机械复制 AI SDK 的 Node/Edge 模块拆分；Rust 侧保留统一 token-provider 抽象。

## Chosen design

### 1. Canonical id 采用 `vertex-maas`

Siumai 的 canonical provider id 采用 `vertex-maas`，同时保留：

- `google-vertex-maas`
- `vertex.maas`

作为 alias，分别对应包名发现路径和 AI SDK provider `name`。

### 2. Registry-layer first-class wrapper，而不是新 provider crate

核心实现落在：

- `siumai-registry/src/registry/factories/vertex_maas.rs`

这层 wrapper 复用 shared OpenAI-compatible runtime，向外暴露统一的：

- chat
- completion
- embedding

能力面。

这样做的原因很直接：

- AI SDK 自己也是基于 `createOpenAICompatible(...)`
- MaaS 本身没有值得单独维护一套 Rust-native 文本 runtime 的额外协议收益
- registry-layer wrapper 更便于和 TogetherAI / DeepInfra 这类“统一 surface + 复用 shared runtime”
  的方案保持一致

### 3. Base URL 语义对齐 AI SDK

`vertex-maas` 的 base URL 优先级现在是：

1. `ctx.base_url`
2. `ctx.project + ctx.location`
3. `GOOGLE_VERTEX_PROJECT + GOOGLE_VERTEX_LOCATION`
4. `location` 默认回落到 `global`

最终拼成：

`https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/endpoints/openapi`

这和 AI SDK `createVertexMaas(...)` 的主语义保持一致。

### 4. Auth 语义改成 Google-first，而不是 api-key-first

最终 auth 优先级现在是：

1. `ctx.resolved_google_token_provider()`
2. 已显式提供的 `Authorization` header
3. `ctx.api_key` 作为静态 Bearer token
4. `feature = gcp` 时自动回落 ADC

关键不是顺序本身，而是 shared compat runtime 不再要求“非空 API key 才允许发请求”。
只要最终 headers 里已经有 `Authorization`，就允许继续走。

### 5. 稳定层 typing 也一起补齐

这次不只补了 `ProviderType::VertexMaas` 的消费层，也顺手把：

- `ProviderType::Vertex`
- `ProviderType::AnthropicVertex`

升成了一等稳定枚举，避免 Google Vertex 原生 wrapper 在 provider catalog、validator、
retry 这一层继续退化成 `Custom(...)`。

### 6. Capability 宣称保持保守，避免“类型面大于审计面”

AI SDK 的 `createVertexMaas(...)` 返回 `OpenAICompatibleProvider`，因此技术上会继承：

- `completionModel()`
- `embeddingModel()`
- `textEmbeddingModel()`
- `imageModel()`

但从当前 `repo-ref/ai` 的直接证据看：

- README 只展示 `generateText` / `streamText`
- examples 只覆盖 text/tool-call
- MaaS provider 的专门测试只校验 base URL、auth wrapper、lazy init

所以本次 Siumai 对齐策略是：

- 明确支持并测试 chat/completion/embedding
- 明确不宣称 image/rerank/speech/transcription
- 用负向回归测试锁住这条边界，等上游把 image 变成真正 audited surface 再放开

## Behavioral changes

- `vertex-maas` 现在是 built-in unified provider surface。
- `Provider::vertex_maas()` / `Siumai::builder().vertex_maas()` 直接可用。
- registry 现在能以 `vertex-maas:{model}` 解析：
  - language model
  - completion model
  - embedding model
- provider catalog 会列出 AI SDK 参考里的 curated MaaS model ids。
- `ProviderType`、retry backoff、默认模型建议、stop sequence 限制现在都能识别
  `vertex-maas`。
- Google Vertex 原生 wrapper 的 `vertex` / `anthropic-vertex` 也不再在稳定层退化成
  `Custom(...)`。

## Validation strategy

- registry contract tests:
  - project/location 优先于 env
  - 默认 `location = global`
  - Google Bearer token 注入
  - completion / embedding family 支持
- public-path parity tests:
  - `Siumai::builder()`
  - `Provider::vertex_maas()`
  - registry handle
- public-surface compile tests:
  - `Provider::vertex_maas()`
  - `Siumai::builder().vertex_maas()`
- stable typing tests:
  - `ProviderType::{Vertex, AnthropicVertex, VertexMaas}`
  - provider catalog native metadata resolution

## Remaining follow-up

- Decide later whether Vertex MaaS deserves a provider-owned crate, or should stay a
  registry-layer unified wrapper permanently.
- Revisit the default-model story for the broader `vertex` provider family so chat/image/embedding
  defaults are documented more explicitly across family-specific entry points.
