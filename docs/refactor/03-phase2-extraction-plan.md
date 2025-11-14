# Phase 2: 多 Crate 拆分执行清单

状态：进行中（核心模块与部分 provider 已迁移）
更新时间：2025-11-13

目标
- 将聚合 crate（当前 `siumai`）中的核心模块按职责拆分到独立 crate：
  - `siumai-core`：traits、types、error、core（ProviderSpec/Context）、execution、streaming、retry、utils
  - `siumai-std-openai`：OpenAI 标准（chat/embedding/image/stream），适配器接口
  - `siumai-std-anthropic`：Anthropic 标准（messages/stream），适配器接口
  - 后续：`siumai-provider-*`（原生与兼容提供商）

已完成
- 工作区 members 加入：`crates/siumai-core`、`crates/siumai-std-openai`、`crates/siumai-std-anthropic`、`crates/siumai-provider-openai`、`crates/siumai-provider-openai-compatible`、`crates/siumai-provider-anthropic`、`crates/siumai-provider-minimaxi`
- 三个基础 crate（core/std-openai/std-anthropic）已承载核心抽象与 OpenAI/Anthropic 标准实现，聚合侧通过桥接统一引用
- openai-compatible 与 openai provider 已具备外部 provider crate + feature 门控的桥接路径，支持在外部/内部实现之间切换

迁移顺序与批次
1) 拆 core（低耦合起步）
   - 从 `siumai/src/error.rs`、`siumai/src/traits/*`、`siumai/src/types/*`、`siumai/src/execution/*` 开始
   - 在 `siumai` 中引入 `siumai-core` 依赖，替换路径导入
   - 验证：`cargo check` features 矩阵

2) 标准 OpenAI 迁移
   - 将 `siumai/src/standards/openai/*` 移至 `siumai-std-openai`
   - 维持 openai-compatible 与标准层的关系（已无 provider 依赖）
   - 验证：`--features openai-compatible` 与 `--features openai`

3) 标准 Anthropic 迁移
   - 在 `siumai-std-anthropic` 中新增 `AnthropicChatStandard` / `AnthropicChatAdapter`，基于 `siumai-core` 的 ChatInput/ChatResult/ChatStreamEventCore 提供 Messages API 的请求/响应/流式转换。
   - 聚合层通过 `std_anthropic` 桥接模块，在启用 `std-anthropic-external` 时使用外部标准实现；默认情况下继续使用聚合内的 Anthropic 标准实现，行为不变。
   - `anthropic` 与 `minimaxi` provider 已经通过桥接接入 Anthropic 标准；`minimaxi` 单开仍能编译。

4) provider 分离
   - openai、anthropic、minimaxi、openai-compatible 按需迁移到 `siumai-provider-*`
   - `siumai` 保留聚合导出与 builder/registry 入口

5) CI 矩阵
   - 已新增 GitHub Actions 工作流，覆盖 openai、openai-compatible、minimaxi 单开及部分 external/std-openai-external 组合。
   - 后续可按需引入 cargo-hack，扩大全特性组合覆盖范围。

命名与兼容性
- 参考 Vercel AI SDK：标准与提供商完全分离；openai-compatible 初期聚合供应商，使用 feature 门控（如 `openrouter`、`deepseek`）。
- 聚合 crate 对外 API 不变，维持稳定性。
