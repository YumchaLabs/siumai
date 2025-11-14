# Phase 1b: OpenAI 标准去 provider 化与 openai-compatible 脱钩

状态：完成（本阶段）
更新时间：2025-11-13

目标
- 移除 `standards/openai/*` 对 `providers::openai::*` 的直接依赖。
- 让 `openai-compatible` 特性不再依赖 `openai` provider。
- 保证以下编译矩阵均可通过：
  - `--features openai-compatible`（仅 OpenAI 兼容层）
  - `--features openai`（仅 OpenAI 原生）
  - `--features minimaxi`（MiniMaxi 单开）
  - 默认 `all-providers`

主要变更
- 新增标准层转换器与工具
  - `siumai/src/standards/openai/transformers.rs`
    - `OpenAiStdRequestTransformer`：Chat/Embedding 请求映射（provider 无关）。
    - `OpenAiStdResponseTransformer`：Chat/Embedding/Image 响应映射。
      - Chat 响应：
        - `cfg(feature = "openai-compatible")` 时复用 openai-compatible 的通用响应转换器；
        - 否则走内置最小映射（不依赖 provider）。
  - `siumai/src/standards/openai/utils.rs`
    - `convert_messages_to_openai_json`、`convert_tools_to_openai_format`、`convert_tool_choice`、`parse_finish_reason` 等工具函数（去除对 provider 类型的依赖）。

- 标准层引用改造
  - `standards/openai/chat.rs` 与 `standards/openai/embedding.rs` 改为使用标准层转换器；
  - Chat 标准的流式转换：
    - `cfg(feature = "openai-compatible")` 启用（复用 openai-compatible SSE 转换）；
    - 否则 `stream = None`，避免强行依赖。

- openai-compatible 脱钩
  - `siumai/Cargo.toml`：`openai-compatible = []`（移除对 `openai` 的临时依赖）。
  - 替换 openai-compatible 内部对 `providers::openai::utils` 的引用为 `standards::openai::utils`：
    - `providers/openai_compatible/transformers.rs`、`providers/openai_compatible/streaming.rs`。
  - 注册与工厂：
    - `registry/mod.rs`：`get_provider_adapter` 与 `register_openai_compatible_providers` 均改为 `cfg(feature = "openai-compatible")`。
    - `registry/factories.rs`：OpenRouter/DeepSeek 工厂改为 `cfg(feature = "openai-compatible")`，避免 openai-only 时报错。

- OpenAI 原生在无 openai-compatible 时的回退
  - `providers/openai/transformers/response.rs`：
    - `cfg(feature = "openai-compatible")` 走 compat 路径；
    - 否则使用最小原生解析（无 SSE）。
  - `providers/openai/transformers/mod.rs`：`stream` 模块在 `openai-compatible` 下才编译导出。
  - `providers/openai/spec.rs`：
    - Chat：
      - Responses API 分支在 `openai-compatible` 下带 `stream`，否则无流式；
      - 非 Responses API：
        - `openai-compatible` 下：复用标准层 `OpenAiChatStandard`；
        - 否则：回退到 provider-native 请求/响应转换，`stream = None`。
    - Embedding：
      - `openai-compatible` 下使用标准层；否则使用 provider-native 请求/响应转换。

- MiniMaxi 去 Anthropic 依赖
  - `siumai/Cargo.toml`：`minimaxi = ["dep:hex"]`（移除 `anthropic`）。
  - 编译依赖由 `standards/anthropic` 提供（此前已完成去 provider 化）。

编译矩阵（本地验证）
- `cargo check -p siumai --no-default-features --features openai-compatible`：通过。
- `cargo check -p siumai --no-default-features --features openai`：通过（无流式降级）。
- `cargo check -p siumai --no-default-features --features minimaxi`：通过。
- `cargo check -p siumai`（默认 all-providers）：通过。

兼容性与注意事项
- 当未启用 `openai-compatible` 时：
  - OpenAI 原生 Chat/Embedding 仍可用；
  - 流式（SSE）路径关闭；
  - Responses API 的流式同样关闭（非破坏性：返回 `stream = None`）。
- Chat 标准中的 `StandardOpenAiProviderAdapter` 与流式代码仅在 `openai-compatible` 下编译，避免额外依赖。

后续工作（Phase 2 提前规划）
- 按蓝图抽离多 crate：`siumai-core`、`siumai-std-openai`、`siumai-std-anthropic`、`siumai-provider-*`、`siumai` 聚合。
- 将 openai-compatible 完全依赖 `siumai-std-openai`，不再引用聚合 crate 内部模块。
- 完善 OpenAI 标准层的原生响应/流式映射（减少针对 compat 的分支）。

