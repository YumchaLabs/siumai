# Phase 2c: OpenAI Chat 流式（SSE）外移与桥接

状态: 完成（最小可用，事件子集）
更新时间: 2025-11-13

## 目标
- 将 OpenAI Chat SSE 事件转换从聚合 crate 内部标准迁移到 `siumai-std-openai`。
- 在 `siumai-core` 提供最小的流式事件抽象与转换 trait，避免外部标准依赖聚合层。
- 在 providers（openai/openai-compatible）通过桥接将外部标准的核心事件转换为聚合层的 `ChatStreamEvent`，保持行为一致。

## 关键改动
- Core
  - `crates/siumai-core/src/execution/streaming.rs`：定义 `ChatStreamEventCore` 与 `ChatStreamEventConverterCore`（SSE → 核心事件）。
- std-openai
  - `crates/siumai-std-openai/src/openai/chat.rs`
    - `OpenAiChatAdapter` 增加 `transform_sse_event` 钩子（默认空实现）。
    - 新增 `create_stream_converter()` 与内部 `OpenAiChatStreamConverter`，解析 OpenAI Chat Completions SSE：
      - 支持 `choices[].delta.content` → ContentDelta
      - 支持 `choices[].delta.tool_calls[].function.{name,arguments}` → ToolCallDelta
      - 支持顶层 `usage` → UsageUpdate
      - 未识别的片段作为 `Custom(openai:unknown_chunk, data)` 透传
- Providers 桥接
  - OpenAI：`siumai/src/providers/openai/spec.rs:121`
    - 启用 `std-openai-external` 时：
      - Chat 请求/响应走外部标准（已在前步完成）
      - 新增 SSE 桥接，将 `ChatStreamEventCore` 映射为聚合层 `ChatStreamEvent`，`StreamStart` 注入最小 `ResponseMetadata`
  - OpenAI-compatible：`siumai/src/providers/openai_compatible/spec.rs:207,220`
    - 同样在启用 `std-openai-external` 时桥接 SSE

## 编译矩阵（均通过）
- `cargo check -p siumai`
- `cargo check -p siumai --no-default-features --features minimaxi`
- `cargo check -p siumai --no-default-features --features openai-compatible`
- `cargo check -p siumai --no-default-features --features openai-compatible,std-openai-external`
- `cargo check -p siumai --no-default-features --features openai,std-openai-external`

## 注意事项
- 外部标准当前不直接发 `StreamEnd` 事件；最终 `StreamEnd` 由聚合层的处理器累积产生（与现有行为一致）。
- `ResponseMetadata` 在 `StreamStart` 中以最小结构填充（provider 名称）；如需更多字段，可后续在桥接处注入。
- 适配器 `transform_sse_event` 可用于对兼容厂商（如思考内容字段、工具增量等）做增量归一化。

## 后续
- 将 `Rerank` 标准外移至 `siumai-std-openai`，并以相同模式桥接。
- 收敛内部标准未使用的 import/warn，适时清理。
- CI 引入 cargo-hack 检查更多组合（含 `std-openai-external` + 外移模块）。

