# Provider Headers 策略与别名（OpenAI-Compatible / OpenAI）

状态: 进行中（保守策略已落地 OpenRouter；其余按需补强）

## 目标
- 统一各 Provider 的请求头构建入口（外部 helpers），避免分散在聚合层。
- 遵循“保守不注入”的策略：仅对已提供的头做别名复制，不擅自新增业务头。
- 提供清晰的扩展点与测试样例，便于后续按官方文档增强。

## 现状
- OpenAI-Compatible 外部 helpers（`siumai-provider-openai-compatible`）
  - `build_json_headers_with_provider(provider_id, api_key, http_extra, config_headers, adapter_headers)`
  - 基线（所有 provider）：
    - Content-Type: application/json
    - Accept: application/json
    - Authorization: Bearer <api_key>（若任何来源已含 Authorization，则不注入）
    - 合并顺序：defaults → http_extra → config_headers → adapter_headers（后者覆盖前者）
  - 已实现别名策略：
    - OpenRouter：若存在 `Referer` 且缺少 `HTTP-Referer`，复制 `Referer → HTTP-Referer`；不注入 X-Title 默认值。
  - 其余 provider（DeepSeek / SiliconFlow / Groq 等）：不做别名注入，仅透传已有头部。

- OpenAI 外部 helpers（`siumai-provider-openai`）
  - `build_openai_json_headers(api_key, organization, project, http_extra)`
  - 注入：Content-Type、Authorization、OpenAI-Organization、OpenAI-Project，并合并 `http_extra`。

## 扩展策略（建议）
- “别名复制优先”：仅在已有标准头场景下做 alias 复制（例如 Provider 要求的别名头），减少信息泄露风险。
- “不注入默认值”：除非官方明确要求，否则不生成新头。
- “就近配置”：Prefer 从 adapter.custom_headers / http_extra 读取用户显式提供值。

## 测试策略
- 新增 `siumai/tests/providers/openai_compatible_headers_alias_test.rs` 覆盖：
  - OpenRouter Referer → HTTP-Referer 复制
  - 无 Referer 不注入 HTTP-Referer
  - DeepSeek/SiliconFlow/Groq 不做别名注入（透传验证）

## 后续计划
- 待官方文档确认后，补齐 DeepSeek / SiliconFlow / Groq 等的“有据可依”的别名或规范化处理，并为每条规则新增对应用例。
- 将更多无状态的 headers 构建逻辑集中于外部 helpers，聚合侧仅桥接。

