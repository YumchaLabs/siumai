# Phase 2a: OpenAI Image 标准外移与桥接

状态: 完成（最小可用）
更新时间: 2025-11-13

## 目标

- 将 OpenAI Image 标准从聚合 crate（`siumai`）外移到独立标准 crate（`crates/siumai-std-openai`）。
- 保持对外 API 与行为稳定，允许通过 feature 切换内置/外部实现。
- 不引入对 provider 的反向依赖，标准仅依赖 `siumai-core`。

## 关键改动

- 新增最小抽象（core）
  - `crates/siumai-core/src/execution/image.rs`
    - `ImageHttpBody`（JSON 或 multipart）
    - `ImageRequestTransformer`（transform_image/edit/variation）
    - `ImageResponseTransformer`（transform_image_response）
  - `crates/siumai-core/src/utils/mime.rs`（迁移 guess_mime 系列，`infer` 作为 workspace 依赖）
  - 暴露模块：`pub mod execution; pub mod utils;`

- 外部标准 crate（std-openai）
  - `crates/siumai-std-openai/src/openai/image.rs`：从 `siumai/src/standards/openai/image.rs` 提取并改造为依赖 `siumai-core` 的实现。
  - Cargo 依赖：`siumai-core`, `reqwest`（workspace）。

- 聚合侧桥接（零入侵引用路径）
  - `siumai/src/std_openai.rs` 提供统一入口：
    - `openai::image`：在启用 `std-openai-external` 时指向外部 crate，否则回落到内部实现。
    - `openai::chat/embedding/rerank`：仍指向内部标准，后续迁移。

- 适配器桥接（provider 侧）
  - 由于 `core` 的 Image 标准仅暴露最小 trait，聚合层的 `ImageTransformers` 仍使用老的 `RequestTransformer/ResponseTransformer`。
  - 在 `openai_compatible/spec.rs` 与 `openai/spec.rs` 内按 feature（`std-openai-external`）增加桥接包装器：
    - 将 `Arc<dyn ImageRequestTransformer>`/`Arc<dyn ImageResponseTransformer>` 包装为聚合侧 `RequestTransformer`/`ResponseTransformer`。
    - 非外部模式下仍直接使用内部标准返回的聚合侧 transformers。

## 校验矩阵（已通过）

```
cargo check -p siumai                                 # 默认 all-providers
cargo check -p siumai --no-default-features --features minimaxi
cargo check -p siumai --no-default-features --features openai-compatible
cargo check -p siumai --no-default-features --features openai-compatible,std-openai-external
cargo check -p siumai --no-default-features --features openai,std-openai-external
```

## 注意事项

- 仅 Image 标准已外移；Chat/Embedding/Rerank 仍在内部标准（`siumai/src/standards/openai/*`）。桥接层在 `std-openai-external` 开启时仅替换 `openai::image` 子模块，其余保持内部实现。
- 迁移 `utils::mime` 至 core 后，`siumai/src/utils/mime.rs` 改为重导出 core 实现，避免重复逻辑。
- `siumai-core` 暂未引入 Chat/Embedding 等通用 transformers，后续随对应标准外移再补齐（遵循“小步快跑，最小暴露”原则）。

## 后续计划（Phase 2b/2c）

1) 外移 OpenAI Embedding 标准到 `siumai-std-openai`
   - 在 core 增补 Embedding 专用最小 trait（避免一次性引入通用 transformers）。
   - 复用现有桥接策略（`std_openai.rs` 局部切换 + provider 层 wrapper）。

2) 外移 OpenAI Chat 标准（含 SSE 映射）
   - 先最小可用：非流式 + SSE 基础事件，保持 `openai-compatible` 门控语义。
   - 补齐工具调用增量事件（必要时以 Custom 事件包裹）。

3) 完成 `openai-compatible` 对 `openai` provider 的零依赖
   - 使 openai-compatible 仅依赖 `siumai-std-openai` + `siumai-core`。

4) CI 编译矩阵
   - 引入 `cargo-hack` 检查常见特性组合。

