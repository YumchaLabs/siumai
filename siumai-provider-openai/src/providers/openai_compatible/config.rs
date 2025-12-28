//! OpenAI-Compatible Providers Configuration
//!
//! This module contains centralized configuration for all OpenAI-compatible providers.
//! Inspired by Cherry Studio's provider configuration system.

use crate::standards::openai::compat::provider_registry::{ProviderConfig, ProviderFieldMappings};
use std::collections::HashMap;

/// Get all built-in provider configurations
pub fn get_builtin_providers() -> HashMap<String, ProviderConfig> {
    let mut providers = HashMap::new();

    // DeepSeek - Advanced reasoning models
    providers.insert(
        "deepseek".to_string(),
        ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec!["reasoning_content".to_string(), "thinking".to_string()],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "reasoning".to_string(),
            ],
            default_model: Some("deepseek-chat".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // SiliconFlow - Hosted AI models
    providers.insert(
        "siliconflow".to_string(),
        ProviderConfig {
            id: "siliconflow".to_string(),
            name: "SiliconFlow".to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec!["reasoning_content".to_string(), "thinking".to_string()],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "reasoning".to_string(),
                "rerank".to_string(),
                "embedding".to_string(),
                "image_generation".to_string(),
            ],
            default_model: Some("deepseek-ai/DeepSeek-V3".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // OpenRouter - Model aggregator
    providers.insert(
        "openrouter".to_string(),
        ProviderConfig {
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "reasoning".to_string(),
                // OpenRouter exposes the OpenAI-style `/embeddings` endpoint.
                // Image generation is not documented as an OpenAI-compatible endpoint (as-of 2025-12-24).
                "embedding".to_string(),
            ],
            default_model: None,
            supports_reasoning: true, // Some models support reasoning
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Together AI - Open source models
    providers.insert(
        "together".to_string(),
        ProviderConfig {
            id: "together".to_string(),
            name: "Together AI".to_string(),
            base_url: "https://api.together.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            // Together documents OpenAI-style `/embeddings` and `/images/generations` endpoints.
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
                "image_generation".to_string(),
            ],
            default_model: Some("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Fireworks AI - Fast inference
    providers.insert(
        "fireworks".to_string(),
        ProviderConfig {
            id: "fireworks".to_string(),
            name: "Fireworks AI".to_string(),
            base_url: "https://api.fireworks.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("accounts/fireworks/models/llama-v3p1-8b-instruct".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // GitHub Copilot - AI-powered coding assistant
    providers.insert(
        "github_copilot".to_string(),
        ProviderConfig {
            id: "github_copilot".to_string(),
            name: "GitHub Copilot".to_string(),
            base_url: "https://api.githubcopilot.com".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("gpt-4".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Perplexity - Search-enhanced AI
    providers.insert(
        "perplexity".to_string(),
        ProviderConfig {
            id: "perplexity".to_string(),
            name: "Perplexity".to_string(),
            base_url: "https://api.perplexity.ai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string()],
            default_model: Some("llama-3.1-sonar-small-128k-online".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Groq - Ultra-fast inference
    // Docs: https://console.groq.com/docs/openai
    providers.insert(
        "groq".to_string(),
        ProviderConfig {
            id: "groq".to_string(),
            name: "Groq".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string()],
            default_model: Some("llama-3.1-8b-instant".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Mistral AI - European AI
    // Docs: https://docs.mistral.ai/api/
    // MiniMaxi - OpenAI-compatible M2 text API
    providers.insert(
        "minimaxi".to_string(),
        ProviderConfig {
            id: "minimaxi".to_string(),
            name: "MiniMaxi".to_string(),
            base_url: "https://api.minimaxi.com/v1".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec![
                    "reasoning_content".to_string(),
                    "thinking".to_string(),
                    "thoughts".to_string(),
                ],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec!["tools".to_string(), "reasoning".to_string()],
            default_model: Some("MiniMax-M2".to_string()),
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );
    providers.insert(
        "mistral".to_string(),
        ProviderConfig {
            id: "mistral".to_string(),
            name: "Mistral AI".to_string(),
            base_url: "https://api.mistral.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            // Mistral OpenAPI spec documents `/v1/embeddings` (as-of 2025-12-24).
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
            ],
            default_model: Some("mistral-large-latest".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Cohere - Enterprise AI
    // Docs: https://docs.cohere.com/
    providers.insert(
        "cohere".to_string(),
        ProviderConfig {
            id: "cohere".to_string(),
            name: "Cohere".to_string(),
            base_url: "https://api.cohere.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "embedding".to_string()],
            default_model: Some("command-r-plus".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Zhipu AI (GLM) - Chinese AI
    // Docs: https://open.bigmodel.cn/dev/api
    providers.insert(
        "zhipu".to_string(),
        ProviderConfig {
            id: "zhipu".to_string(),
            name: "Zhipu AI".to_string(),
            base_url: "https://open.bigmodel.cn/api/paas/v4".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("glm-4-plus".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Moonshot AI - Chinese AI with long context
    // Reference: https://platform.moonshot.cn/docs/intro
    providers.insert(
        "moonshot".to_string(),
        ProviderConfig {
            id: "moonshot".to_string(),
            name: "Moonshot AI".to_string(),
            base_url: "https://api.moonshot.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("kimi-k2-0905-preview".to_string()), // Updated to latest K2 model
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Baichuan AI - Chinese AI
    providers.insert(
        "baichuan".to_string(),
        ProviderConfig {
            id: "baichuan".to_string(),
            name: "Baichuan AI".to_string(),
            base_url: "https://api.baichuan-ai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string()],
            default_model: Some("Baichuan2-Turbo".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // 01.AI (Yi) - Chinese AI
    // Docs: https://platform.lingyiwanwu.com/docs
    providers.insert(
        "yi".to_string(),
        ProviderConfig {
            id: "yi".to_string(),
            name: "01.AI".to_string(),
            base_url: "https://api.lingyiwanwu.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("yi-large".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Doubao (ByteDance) - Chinese AI
    providers.insert(
        "doubao".to_string(),
        ProviderConfig {
            id: "doubao".to_string(),
            name: "Doubao".to_string(),
            base_url: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
            field_mappings: ProviderFieldMappings {
                thinking_fields: vec!["thinking".to_string()],
                content_field: "content".to_string(),
                tool_calls_field: "tool_calls".to_string(),
                role_field: "role".to_string(),
            },
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "reasoning".to_string(),
            ],
            default_model: Some("doubao-pro-4k".to_string()),
            supports_reasoning: true, // Doubao supports thinking
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Qwen (Alibaba) - Chinese AI
    // Docs: https://help.aliyun.com/zh/dashscope/developer-reference/compatible-openai
    providers.insert(
        "qwen".to_string(),
        ProviderConfig {
            id: "qwen".to_string(),
            name: "Qwen".to_string(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("qwen-plus".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Groq (OpenAI-compatible) - Ultra-fast inference
    providers.insert(
        "groq_openai_compatible".to_string(),
        ProviderConfig {
            id: "groq_openai_compatible".to_string(),
            name: "Groq (OpenAI Compatible)".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("llama-3.3-70b-versatile".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // xAI (OpenAI-compatible) - Grok models
    // Docs: https://x.ai/api
    providers.insert(
        "xai_openai_compatible".to_string(),
        ProviderConfig {
            id: "xai_openai_compatible".to_string(),
            name: "xAI (OpenAI Compatible)".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("grok-beta".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Nvidia - GPU-accelerated AI models
    providers.insert(
        "nvidia".to_string(),
        ProviderConfig {
            id: "nvidia".to_string(),
            name: "Nvidia".to_string(),
            base_url: "https://integrate.api.nvidia.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("meta/llama-3.1-405b-instruct".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Hyperbolic - Fast inference platform
    providers.insert(
        "hyperbolic".to_string(),
        ProviderConfig {
            id: "hyperbolic".to_string(),
            name: "Hyperbolic".to_string(),
            base_url: "https://api.hyperbolic.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // StepFun - Chinese AI platform
    providers.insert(
        "stepfun".to_string(),
        ProviderConfig {
            id: "stepfun".to_string(),
            name: "StepFun".to_string(),
            base_url: "https://api.stepfun.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("step-1v-8k".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // MiniMax - Chinese AI with advanced capabilities
    providers.insert(
        "minimax".to_string(),
        ProviderConfig {
            id: "minimax".to_string(),
            name: "MiniMax".to_string(),
            base_url: "https://api.minimax.chat/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("abab6.5s-chat".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Jina AI - Embedding and multimodal AI
    providers.insert(
        "jina".to_string(),
        ProviderConfig {
            id: "jina".to_string(),
            name: "Jina AI".to_string(),
            base_url: "https://api.jina.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string(), "rerank".to_string()],
            default_model: Some("jina-embeddings-v2-base-en".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // GitHub Models - Free AI models from GitHub
    providers.insert(
        "github".to_string(),
        ProviderConfig {
            id: "github".to_string(),
            name: "GitHub Models".to_string(),
            base_url: "https://models.github.ai/inference/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("gpt-4o-mini".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Infini AI - Chinese cloud AI platform
    providers.insert(
        "infini".to_string(),
        ProviderConfig {
            id: "infini".to_string(),
            name: "Infini AI".to_string(),
            base_url: "https://cloud.infini-ai.com/maas/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
            ],
            default_model: Some("deepseek-chat".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // 302.AI - Multi-model AI platform
    providers.insert(
        "302ai".to_string(),
        ProviderConfig {
            id: "302ai".to_string(),
            name: "302.AI".to_string(),
            base_url: "https://api.302.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "image_generation".to_string(),
            ],
            default_model: Some("gpt-4o-mini".to_string()),
            supports_reasoning: false,
            api_key_env: Some("AI302_API_KEY".to_string()),
            api_key_env_aliases: Vec::new(),
        },
    );

    // AiHubMix - AI model aggregation platform
    providers.insert(
        "aihubmix".to_string(),
        ProviderConfig {
            id: "aihubmix".to_string(),
            name: "AiHubMix".to_string(),
            base_url: "https://aihubmix.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "image_generation".to_string(),
            ],
            default_model: Some("gpt-4o-mini".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // VoyageAI - Professional embedding provider
    providers.insert(
        "voyageai".to_string(),
        ProviderConfig {
            id: "voyageai".to_string(),
            name: "VoyageAI".to_string(),
            base_url: "https://api.voyageai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["embedding".to_string(), "rerank".to_string()],
            default_model: Some("voyage-3".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // ModelScope - Alibaba's model community
    providers.insert(
        "modelscope".to_string(),
        ProviderConfig {
            id: "modelscope".to_string(),
            name: "ModelScope".to_string(),
            base_url: "https://api-inference.modelscope.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("qwen2.5-72b-instruct".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Hunyuan - Tencent's AI platform
    providers.insert(
        "hunyuan".to_string(),
        ProviderConfig {
            id: "hunyuan".to_string(),
            name: "Hunyuan".to_string(),
            base_url: "https://api.hunyuan.cloud.tencent.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("hunyuan-pro".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Baidu Cloud - Baidu's AI cloud platform
    providers.insert(
        "baidu_cloud".to_string(),
        ProviderConfig {
            id: "baidu_cloud".to_string(),
            name: "Baidu Cloud".to_string(),
            base_url: "https://qianfan.baidubce.com/v2".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("ernie-4.0-8k".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Poe - Quora's AI platform
    providers.insert(
        "poe".to_string(),
        ProviderConfig {
            id: "poe".to_string(),
            name: "Poe".to_string(),
            base_url: "https://api.poe.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("gpt-4o".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Tencent Cloud TI - Tencent's cloud AI platform
    providers.insert(
        "tencent_cloud_ti".to_string(),
        ProviderConfig {
            id: "tencent_cloud_ti".to_string(),
            name: "Tencent Cloud TI".to_string(),
            base_url: "https://api.lkeap.cloud.tencent.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("hunyuan-lite".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // Xirang - China Telecom's AI platform
    providers.insert(
        "xirang".to_string(),
        ProviderConfig {
            id: "xirang".to_string(),
            name: "Xirang".to_string(),
            base_url: "https://wishub-x1.ctyun.cn/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("xirang-chat".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // PPIO - AI infrastructure platform
    providers.insert(
        "ppio".to_string(),
        ProviderConfig {
            id: "ppio".to_string(),
            name: "PPIO".to_string(),
            base_url: "https://api.ppinfra.com/v3/openai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("gpt-4o-mini".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    // OcoolAI - Cool AI platform
    providers.insert(
        "ocoolai".to_string(),
        ProviderConfig {
            id: "ocoolai".to_string(),
            name: "OcoolAI".to_string(),
            base_url: "https://api.ocoolai.com/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("gpt-4o-mini".to_string()),
            supports_reasoning: false,
            api_key_env: None,
            api_key_env_aliases: Vec::new(),
        },
    );

    providers
}

/// Get provider configuration by ID
pub fn get_provider_config(provider_id: &str) -> Option<ProviderConfig> {
    get_builtin_providers().get(provider_id).cloned()
}

/// List all available provider IDs
pub fn list_provider_ids() -> Vec<String> {
    get_builtin_providers().keys().cloned().collect()
}

/// Check if a provider supports a specific capability
pub fn provider_supports_capability(provider_id: &str, capability: &str) -> bool {
    if let Some(config) = get_provider_config(provider_id) {
        config.capabilities.contains(&capability.to_string())
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_providers() {
        let providers = get_builtin_providers();

        // Should have all major providers
        assert!(providers.contains_key("deepseek"));
        assert!(providers.contains_key("openrouter"));
        assert!(providers.contains_key("together"));
        assert!(providers.contains_key("fireworks"));

        // DeepSeek should support reasoning
        let deepseek = providers.get("deepseek").unwrap();
        assert!(deepseek.supports_reasoning);
        assert!(deepseek.capabilities.contains(&"reasoning".to_string()));

        // All providers should have valid URLs
        for (id, config) in &providers {
            assert!(
                !config.base_url.is_empty(),
                "Provider {} has empty base_url",
                id
            );
            assert!(
                config.base_url.starts_with("https://"),
                "Provider {} has invalid URL",
                id
            );
        }
    }

    #[test]
    fn test_provider_capabilities() {
        assert!(provider_supports_capability("deepseek", "reasoning"));
        assert!(provider_supports_capability("openrouter", "tools"));
        assert!(provider_supports_capability("openrouter", "embedding"));
        assert!(provider_supports_capability("perplexity", "tools"));
        assert!(provider_supports_capability("siliconflow", "rerank"));
        assert!(provider_supports_capability("siliconflow", "embedding"));
        assert!(provider_supports_capability(
            "siliconflow",
            "image_generation"
        ));
        assert!(provider_supports_capability("together", "embedding"));
        assert!(provider_supports_capability("together", "image_generation"));
        assert!(provider_supports_capability("mistral", "embedding"));
        assert!(!provider_supports_capability("groq", "vision"));
    }

    #[test]
    fn test_provider_config_retrieval() {
        let config = get_provider_config("deepseek").unwrap();
        assert_eq!(config.id, "deepseek");
        assert_eq!(config.name, "DeepSeek");

        assert!(get_provider_config("nonexistent").is_none());
    }
}
