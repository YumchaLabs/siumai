# Phase 2d: Provider Crates 提取计划（OpenAI / OpenAI-Compatible / Anthropic / MiniMaxi / Groq / xAI)

状态: 进行中（OpenAI / OpenAI-Compatible / Groq / xAI 已接入外部实现路径）
更新时间: 2025-11-16

## 目标
- 将聚合 crate `siumai` 中 `providers/*` 迁出，独立为 provider 子 crate：
  - `siumai-provider-openai`
  - `siumai-provider-openai-compatible`
  - `siumai-provider-anthropic`
  - `siumai-provider-minimaxi`
  - `siumai-provider-xai`
  - `siumai-provider-groq`
- Provider 仅依赖 `siumai-core` + 相关标准 crate（`siumai-std-openai`、`siumai-std-anthropic`）。
- 聚合 crate 通过 feature 门控选择内置实现或外部 provider crate（与 `std-openai-external` 相同策略）。

## 当前进展
- 新增并接入若干 provider crate：
  - `crates/siumai-provider-openai`：
    - VERSION + 标记类型
    - 已外提最小 adapter：`adapter::OpenAiStandardAdapter`（实现统一 ProviderAdapter 接口）
    - 已外提无状态路由 helpers：`chat_path/embedding_path/image_generation_path/image_edit_path/image_variation_path`
    - 已外提 OpenAI JSON 请求头构建：`build_openai_json_headers`
    - 已外提 Responses API 相关 SSE 事件名常量：`RESPONSES_EVENT_*`
    - 聚合侧 `OpenAiSpec` 在启用 `provider-openai-external` 时通过 helpers 生成 URL/Headers 与 Responses 判定
  - `crates/siumai-provider-openai-compatible`：
    - adapter/types/registry/helpers 已抽出
    - helpers 内提供 `build_json_headers_with_provider`（带 provider_id），为 OpenRouter 提供 Referer→HTTP-Referer 保守别名策略，其余 provider 仅透传
    - 聚合侧 Spec 与 Client 在启用 `provider-openai-compatible-external` 时委托外部实现
  - `crates/siumai-provider-anthropic`：
    - VERSION + Marker + `AnthropicCoreSpec`（基于 `siumai-std-anthropic` 的 CoreProviderSpec 实现）
    - 提供 `build_anthropic_json_headers` helper，聚合层在启用 `provider-anthropic-external` 时委托该 helper 构建 headers。
  - `crates/siumai-provider-minimaxi`（VERSION + Marker，占位，待后续迁移）
  - `crates/siumai-provider-groq`：
    - 提供 `GroqCoreSpec`（基于 `siumai-std-openai` 的 CoreProviderSpec 实现），在 core 层复用 OpenAI Chat 标准。
    - 提供 `headers::build_groq_json_headers`，聚合层 `ProviderHeaders::groq` 在启用 `provider-groq-external` 时委托该 helper 构建 headers。
    - 在 Chat 路径中通过 `GroqOpenAiChatAdapter` 从 `ChatInput::extra["groq_extra_params"]` 注入 Groq-specific JSON 字段。
  - `crates/siumai-provider-xai`：
    - 提供 `XaiCoreSpec`（基于 `siumai-std-openai` 的 CoreProviderSpec 实现），在 core 层复用 OpenAI Chat 标准。
    - 提供 `headers::build_xai_json_headers`，聚合层 `ProviderHeaders::xai` 在启用 `provider-xai-external` 时委托该 helper 构建 headers。

- 标准层与 provider 桥接：
  - OpenAI：`siumai-std-openai` 通过 `std_openai` 模块桥接到聚合层；OpenAI / openai-compatible provider 在启用 `std-openai-external` 时使用外部标准实现。
  - Anthropic：`siumai-std-anthropic` 提供 `AnthropicChatStandard` 与流式转换（输出 `ChatStreamEventCore`）；聚合层 `AnthropicEventConverter` 在启用 `std-anthropic-external` 时委托 std-anthropic 进行 SSE 解析，并映射为最终 `ChatStreamEvent`。
- 工作区加入：`Cargo.toml: [workspace].members`
- 聚合 crate 新增外部特性门控：
  - `provider-openai-external` / `provider-openai-compatible-external` / `provider-anthropic-external` / `provider-minimaxi-external`

## 迁移策略（分阶段）
1) 路径桥接（与 std-openai 桥接一致）
   - 在 `siumai/src/providers/<name>/mod.rs` 加入：
     - `#[cfg(feature = "provider-<name>-external")] pub use siumai_provider_<name>::*;`
     - `#[cfg(not(...))]` 保留现有内部实现
   - 首先对 openai 与 openai-compatible 实施，验证 cargo check 矩阵。

2) 类型/能力抽离
   - 将各 provider 侧通用的 transformers/client/spec 的核心依赖改为 `siumai-core` + `std-<standard>`（当前大部分已通过标准层进行），逐步去除对聚合 crate 其他模块的引用。

3) 批量迁移实现文件
   - 将 `siumai/src/providers/<name>/*` 的实现文件移动到对应 provider crate，按模块组织。
   - 聚合侧保留薄桥与 re-export，确保对外 API 稳定。

4) CI 编译矩阵
   - 引入 `cargo-hack` 覆盖：
     - provider 内置/外部切换
     - 与 `std-openai-external` 的组合
     - 典型 provider 组合（openai-only、openai-compatible-only、minimaxi-only 等）

## 风险与回退
- 桥接层不改变外部 API；失败可关闭 `provider-*-external` 特性回退到内置实现。
- 迁移初期 provider crate 仅作为空壳，逐步搬运，保证每一步可编译。

## 下一步
- Anthropic / MiniMaxi provider 外提
  - 为 `siumai-provider-anthropic` 与 `siumai-provider-minimaxi` 按当前模式添加 adapter/helpers/scaffolding，并在聚合层加上 `provider-*-external` 桥接。
- OpenAI / OpenAI-Compatible 外提收尾
  - 如 Responses API 未来增加更多事件/标志位，将常量与判定逻辑继续下沉到 `siumai-provider-openai::helpers`，聚合层仅桥接调用。
  - 根据官方文档，为 DeepSeek/SiliconFlow/Groq 等补充“仅复制已提供值”的 header 别名，并为每条规则补充 tests。
- CI 与示例
  - 在 CI 中运行特性矩阵（见 scripts/check-feature-matrix.sh 或 GitHub Actions 工作流），确保外部/内部实现与标准外部化组合均可编译。
  - 增加 examples 展示启用 external provider/standard 组合时的使用方式。

## openai-compatible 特定 header 策略（保守增强）
- 在外部 helpers 中新增 `build_json_headers_with_provider`（带 provider_id）：
  - OpenRouter：若仅提供了标准 `Referer` 而未提供 `HTTP-Referer`，自动复制为 `HTTP-Referer`，其余不注入默认值（避免信息泄露）。
  - 其余提供商：维持通用策略（Content-Type/Accept/Authorization + 合并 http_extra/config/adapter）。
