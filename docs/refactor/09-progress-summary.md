# Phase 2 进度快照（截至 0.11.0-beta.3 重构）

状态：进行中
更新时间：2025-11-16

## 已完成（里程碑）

- 架构分层
  - 建立「core + standards + providers + 聚合」多 crate 拆分骨架。
  - `siumai-core`：承载 error/traits/types/core/execution/streaming/retry/utils 的最小集合。
  - `siumai-std-openai`、`siumai-std-anthropic`：承载 OpenAI/Anthropic 标准实现，完全去 provider 化。

- OpenAI 标准与 provider 重构
  - OpenAI 标准迁移至 `siumai-std-openai`，聚合侧统一通过 `std_openai` 桥接调用。
  - OpenAI 标准与 OpenAI provider 解耦：标准不再引用 `providers::openai::*`，仅依赖 core。
  - OpenAI provider 外提至 `siumai-provider-openai`：
    - `OpenAiStandardAdapter` 统一实现 ProviderAdapter 接口。
    - 无状态路由 helpers（chat/embedding/image generation/edit/variation）。
    - OpenAI JSON headers 构建函数 `build_openai_json_headers`。
    - Responses API SSE 事件名常量 `RESPONSES_EVENT_*`。
  - 聚合侧 `OpenAiSpec` 在启用 `provider-openai-external` 时通过上述 helpers 生成 URL/Headers 与 Responses 判定，未启用特性则保持原逻辑。
  - 聚合层通用 headers 构建入口 `ProviderHeaders::openai` 已在启用 `provider-openai-external` 时统一委托给 `siumai-provider-openai::helpers::build_openai_json_headers`，实现“就近配置”。

- Anthropic 标准与 MiniMaxi
  - 在聚合内完成 Anthropic 标准去 provider 化，并接入 core 的统一事件模型（ChatStreamEventCore）。
  - `siumai-std-anthropic`：新增 `AnthropicChatStandard` / `AnthropicChatAdapter`，在 std crate 内提供 Anthropic Chat 请求/响应/流式转换（输出 ChatStreamEventCore）。
  - 聚合层 `AnthropicEventConverter` 在启用 `std-anthropic-external` 时使用 std-anthropic 的流式转换结果，并映射为最终的 `ChatStreamEvent`；默认路径仍使用原有实现，行为不变。
  - `minimaxi` feature 移除对 `openai`/`anthropic` provider 的硬依赖，通过桥接使用 Anthropic 标准与 core；`minimaxi` 单开可编译。

- FinishReason 抽象统一
  - 在 `siumai-core` 中新增 `FinishReasonCore`（Stop/Length/ContentFilter/ToolCalls/Other），作为标准层的统一结束原因枚举。
  - OpenAI/Anthropic 标准的 `parse_finish_reason` 先解析为 `FinishReasonCore`，再在聚合层映射为现有的 `crate::types::FinishReason`，保持对外 API 不变。

- OpenAI-Compatible 重构
  - `openai-compatible` 不再依赖 `openai` provider，仅依赖 `std-openai` 标准。
  - 抽取 `siumai-provider-openai-compatible`：
    - adapter/types/registry/helpers 迁移到外部 crate。
    - `build_json_headers_with_provider` 提供 provider 级 header 策略：
      - OpenRouter：若存在 `Referer` 且缺少 `HTTP-Referer`，复制 Referer→HTTP-Referer；其余不注入默认值。
      - DeepSeek/SiliconFlow/Groq：仅透传，不做别名注入。
  - 聚合侧在启用 `provider-openai-compatible-external` 时，Spec 与 Client 构建 headers/URL 改用外部 helpers。

- Groq / xAI provider 与 headers 重构
  - `siumai-provider-groq`：
    - 提供 `GroqCoreSpec`，在 core 层基于 OpenAI Chat 标准实现 Groq Chat 行为。
    - 提供 `build_groq_json_headers`，聚合层 `ProviderHeaders::groq` 在启用 `provider-groq-external` 时委托该 helper 构建 headers；未启用时保留原有 `HttpHeaderBuilder` 实现。
    - Chat 路径：`GroqOptions.extra_params` 统一映射至 `ChatInput::extra["groq_extra_params"]`，由 `GroqOpenAiChatAdapter` 在 std/provider 层 merge 到最终 JSON。
  - `siumai-provider-xai`：
    - 提供 `XaiCoreSpec`，在 core 层基于 OpenAI Chat 标准实现 xAI Chat 行为。
    - 提供 `build_xai_json_headers`，聚合层 `ProviderHeaders::xai` 在启用 `provider-xai-external` 时委托该 helper 构建 headers；未启用时保留原有 `HttpHeaderBuilder` 实现。
    - Chat 路径：`XaiOptions` 在聚合层映射为 `ChatInput::extra["xai_search_parameters" | "xai_reasoning_effort"]`，由 XAI-specific OpenAI Chat adapter 注入最终 JSON。

- CI 与特性矩阵
  - 新增 GitHub Actions 工作流 `.github/workflows/ci.yml`：
    - 覆盖 `openai` / `openai-compatible` / `minimaxi` 单开及 external/std-openai-external 的典型组合。
  - 新增本地脚本 `scripts/check-feature-matrix.sh` 用于快速验证常见组合。

-- ProviderOptions 标准化（阶段性完成 → OpenAI/Gemini 已完成）
  - OpenAI：
    - Chat 路径：`ProviderOptions::OpenAi` 通过 `openai_chat_request_to_core_input` 映射到 `ChatInput::extra`（键名 `openai_*`），由 `siumai-std-openai::OpenAiDefaultChatAdapter` 注入最终 JSON；聚合层不再直接基于 OpenAiOptions 拼接 Chat 请求体。
    - Responses 路径：`ResponsesApiConfig` 以及 OpenAI-specific 选项通过 `build_responses_input` 写入 `ResponsesInput::extra`，由 `OpenAiResponsesStandard` 将 `extra` 扁平合并到 `/responses` body；聚合层不再处理 Responses-specific JSON，仅保留 legacy RequestTransformer 作为 fallback。
  - Anthropic：
    - `AnthropicOptions`（thinking / response_format / prompt_caching）通过 `anthropic_like_chat_request_to_core_input` 映射到 `ChatInput::extra`（键名 `anthropic_*`），由 `AnthropicDefaultChatAdapter` 和标准工具函数注入 Messages JSON / `cache_control`。
  - xAI：
    - `XaiOptions` 在聚合层映射为 `ChatInput::extra["xai_search_parameters"]` / `"xai_reasoning_effort"`，由 `XaiOpenAiChatAdapter` 注入 OpenAI-compatible JSON 字段。
  - Groq：
    - `GroqOptions` 包含 typed 字段（`reasoning_effort` / `reasoning_format` / `parallel_tool_calls` / `service_tier`）和逃生口 `extra_params`：
      - 聚合层通过 `groq_chat_request_to_core_input` 将其映射为 `ChatInput::extra["groq_reasoning_effort" | "groq_reasoning_format" | "groq_parallel_tool_calls" | "groq_service_tier" | "groq_extra_params"]`；
      - 在启用 `provider-groq-external` 时，由 `siumai-provider-groq::GroqOpenAiChatAdapter` 注入最终 JSON：typed 字段直接映射为顶层字段（优先级最高），`groq_extra_params` 中的键值在不与 typed 冲突时按原样展开。
  - Gemini：
    - Chat：`GeminiOptions` 通过 `gemini_like_chat_request_to_core_input` 映射为 `ChatInput::extra["gemini_code_execution" | "gemini_search_grounding" | "gemini_file_search" | "gemini_response_mime_type"]`，由 `GeminiDefaultChatAdapter` 注入 `tools` / `generationConfig`；`GeminiSpec::chat_before_send` 仅处理 CustomProviderOptions。
    - Embedding / Image：在启用 `std-gemini-external` 时，始终通过 `GeminiEmbeddingStandard` / `GeminiImageStandard` + `bridge_core_*` 处理；未启用时明确返回 `UnsupportedOperation`，不再静默使用 legacy 聚合层标准。

## 当前可稳定通过的编译组合

- `cargo check -p siumai`（默认 all-providers）。
- `cargo check -p siumai --no-default-features --features openai`。
- `cargo check -p siumai --no-default-features --features openai,provider-openai-external`。
- `cargo check -p siumai --no-default-features --features openai-compatible`。
- `cargo check -p siumai --no-default-features --features openai-compatible,provider-openai-compatible-external`。
- `cargo check -p siumai --no-default-features --features minimaxi`。

（注：标准与 provider 外部化特性 `std-openai-external` 组合亦已验证，通过文档中的脚本/CI 矩阵覆盖。）

## 下一阶段重点计划

1) Anthropic / MiniMaxi provider 外提
- 为 `siumai-provider-anthropic` 与 `siumai-provider-minimaxi` 按 OpenAI 模式补齐：
  - adapter 实现、无状态 helpers（路由/headers），必要时外提 SSE 常量。
  - 聚合侧新增 `provider-anthropic-external`、`provider-minimaxi-external` 桥接，保持对外 API 不变。

2) OpenAI / OpenAI-Compatible 收尾
- 如 Responses API 未来新增事件名或标志位，将常量与判定逻辑统一下沉到 `siumai-provider-openai::helpers`，聚合侧仅桥接。
- 根据官方文档，为 DeepSeek/SiliconFlow/Groq 等补充“仅复制已提供值”的 header 别名策略，并为每条规则新增 tests。

3) 核心抽象进一步收敛
- 按需继续把 execution/transformers 通用 trait/type 下沉到 `siumai-core`，确保 standards/provider 仅依赖 core。

4) CI 与质量
- 在 CI 中引入 `cargo-hack` 或扩展现有矩阵，覆盖 external × std-openai-external × 典型 provider 的组合全集。
- 清理非关键 warning，补全模块级文档与示例，准备对外发布说明（README/CHANGELOG）。
