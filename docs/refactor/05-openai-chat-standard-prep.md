# Phase 2c Prep: OpenAI Chat 标准外移（最小可行，先不含流）

状态: 已落地最小抽象与外部实现骨架（未切换引用）
更新时间: 2025-11-13

## 目标
- 在 core 增加 Chat 最小抽象（请求/响应/用量与参数子集），不依赖聚合层。
- 在 `siumai-std-openai` 实现 Chat 的 request/response 转换，暂不提供流式事件（SSE）。
- 后续通过桥接在 providers 中将外部 Chat 标准适配为聚合层的 `ChatTransformers`（短期 `stream: None`）。

## 当前进展
- Core 最小抽象：`crates/siumai-core/src/execution/chat.rs`
  - `ChatRole/ChatMessageInput/ChatInput`（包含 model/max_tokens/temperature/top_p 等基础参数）
  - `ChatUsage/ChatResult`
  - `ChatRequestTransformer/ChatResponseTransformer`
- 外部标准实现：`crates/siumai-std-openai/src/openai/chat.rs`
  - 映射核心 ChatInput → OpenAI Chat Completions JSON
  - 解析 OpenAI 响应 → ChatResult（content/finish_reason/usage）
  - 适配器接口 `OpenAiChatAdapter`（仅 request/response 钩子；无流）
- 尚未切换聚合层引用（`siumai/src/std_openai.rs` 暂保留 chat 指向内部标准）。

## 切换计划（建议分两步）
1) 非流式切换
   - 在 providers（openai、openai-compatible）侧按 `std-openai-external` 添加桥接包装器：
     - 将 `Arc<dyn ChatRequestTransformer>/Arc<dyn ChatResponseTransformer>` 包装为聚合层 `RequestTransformer/ResponseTransformer`，构造 `ChatTransformers`，`stream: None`。
   - 确保 `openai-compatible` 路径不改变 SSE 语义：在外移流式前，继续使用内部标准以保持流行为。
2) 流式映射外移
   - 在 core 增加最小流事件抽象（增量 delta、tool 调用、用量增量等）；
   - 在 `siumai-std-openai` 实现 SSE → 核心事件；在 providers 桥接到聚合层的 `StreamChunkTransformer` 或提供转换器。

## 编译矩阵
- 非切换阶段（当前）：与 Phase 2a/2b 保持一致，确保不影响现有组合。
- 切换阶段：新增 `std-openai-external` + chat 路径切换的组合编译检查。

## 风险与回退
- 如发现兼容性问题，移除 providers 桥接的 `cfg(std-openai-external)` 即可回退；内部标准仍可用。

