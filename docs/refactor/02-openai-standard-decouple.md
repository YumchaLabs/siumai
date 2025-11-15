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

## Responses API 标准化（补充说明，已接入 Phase 2 架构）

在 Phase 2 的 core/std/provider 拆分中，OpenAI Responses API 也已接入标准层与 core 抽象，具体包括：

- Core 抽象：`crates/siumai-core/src/execution/responses.rs`
  - `ResponsesInput { model, input, extra }`：统一请求入口；
  - `ResponsesResult { output, usage, finish_reason, metadata }`：
    - `output`：标准化为嵌套的 `response` 对象（由 std-openai 负责）；
    - `usage`：包含 `prompt_tokens/completion_tokens/total_tokens`；
    - `finish_reason: Option<FinishReasonCore>`：核心结束原因（`Stop/Length/ToolCalls/ContentFilter/Other`）。

- 标准层实现：`crates/siumai-std-openai/src/openai/responses.rs`
  - Request：
    - `OpenAiResponsesStandard::create_request_transformer` 负责把 `ResponsesInput` 展平为 OpenAI `/responses` JSON body（`model + input[] + extra`）。
  - Response：
    - `OpenAiResponsesStandard::create_response_transformer` 负责：
      - 将顶层 JSON 归一化为 `ResponsesResult.output = raw["response"]`；
      - 从 `usage`（支持 snake/camel 命名）解析出 `ResponsesUsage`；
      - 从 `stop_reason/finish_reason` 解析并规范化 `ResponsesResult.finish_reason`：
        - `"max_tokens"` → `FinishReasonCore::Length`
        - `"tool_use" | "function_call"` → `FinishReasonCore::ToolCalls`
        - `"safety"` → `FinishReasonCore::ContentFilter`
        - 其它字符串 → `FinishReasonCore::Other(String)`。
  - Streaming：
    - `OpenAiResponsesStandard::create_stream_converter` 返回的 `OpenAiResponsesStreamConverter`：
      - 把 Responses SSE 事件映射到 `ChatStreamEventCore`：
        - `response.output_text.delta` → `ContentDelta`
        - `response.tool_call.delta` / `response.function_call.delta` / `response.function_call_arguments.delta` / `response.output_item.added` → `ToolCallDelta`
        - `response.usage` → `UsageUpdate`
        - `response.error` → `Error`
        - 首个事件发出 `StreamStart`；
        - `response.completed` → `StreamEnd { finish_reason: Option<FinishReasonCore> }`（与非流式的 finish_reason 解析规则一致）。

- 聚合层适配：`siumai/src/providers/openai/spec.rs` 与 `.../transformers/response.rs`
  - 在启用 `std-openai-external` 时：
    - 请求：
      - Responses 分支经 `ResponsesRequestBridge` 构造 `ResponsesInput`，交给 `OpenAiResponsesStandard` 构建请求体；
    - 响应：
      - 使用 `OpenAiResponsesStandard` 解析 `ResponsesResult`，然后：
        - `output` → 经 `parse_responses_output` 标准化为 `{ text, tool_calls }`，再映射为聚合层的 `ContentPart`；
        - `usage` → 转为聚合层 `Usage`（并从原始 JSON 补充 reasoning_tokens）；
        - `finish_reason: Option<FinishReasonCore>` → 映射为聚合层 `FinishReason`，若为空则退回旧字符串逻辑。
    - Streaming：
      - Responses 的 SSE 通过 `OpenAiResponsesStreamConverter` 先生成 `ChatStreamEventCore`，再由一个小桥接器使用 `map_core_stream_event_with_provider("openai", ..)` 转为聚合层的 `ChatStreamEvent`。
  - 未启用 `std-openai-external` 时：
    - 保留原有聚合层的 `OpenAiResponsesRequestTransformer` / `OpenAiResponsesResponseTransformer` / `OpenAiResponsesEventConverter` 行为，避免破坏已有调用。

通过这一步，OpenAI Responses API 的「请求/usage/finish_reason/streaming 事件」语义已经在 core/std 层集中管理，聚合层只做轻适配。这也为后续多语言客户端、独立 provider crate（如 `siumai-provider-openai`）共享同一套 Responses 语义打下基础。

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
