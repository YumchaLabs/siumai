//! OpenAI-Compatible Providers Configuration
//!
//! This module contains centralized configuration for all OpenAI-compatible providers.
//! Inspired by Cherry Studio's provider configuration system.

use crate::providers::openai_compatible::ProviderAdapter;
use crate::providers::openai_compatible::fireworks as fireworks_models;
use crate::providers::openai_compatible::mistral as mistral_models;
use crate::providers::openai_compatible::perplexity as perplexity_models;
use crate::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
};
use std::collections::HashMap;
use std::sync::LazyLock;

mod family_defaults;

pub(crate) use family_defaults::{
    get_builtin_provider_family_defaults_map, get_default_embedding_model, get_default_image_model,
    get_default_rerank_model, get_default_speech_model, get_default_transcription_model,
};

#[cfg(test)]
pub(crate) use family_defaults::get_provider_family_defaults_ref;

#[cfg(test)]
use crate::providers::openai_compatible::groq as groq_models;

/// Get all built-in provider configurations
fn build_builtin_providers() -> HashMap<String, ProviderConfig> {
    let mut providers = HashMap::new();

    // DeepSeek - Advanced reasoning models
    providers.insert(
        "deepseek".to_string(),
        ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            // Vercel reference defaults to `https://api.deepseek.com` and uses `/chat/completions`.
            // We follow the same convention for parity; callers can override when needed.
            base_url: "https://api.deepseek.com".to_string(),
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
            api_key_env: Some("DEEPSEEK_API_KEY".to_string()),
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
                // SiliconFlow documents OpenAI-compatible TTS/STT endpoints:
                // `/audio/speech` and `/audio/transcriptions`.
                "speech".to_string(),
                "transcription".to_string(),
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
            // Together documents OpenAI-style `/completions`, `/embeddings`, and
            // `/images/generations` endpoints.
            capabilities: vec![
                "completion".to_string(),
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
                "image_generation".to_string(),
                // Together documents OpenAI-compatible `/audio/speech`
                // and `/audio/transcriptions` endpoints.
                "speech".to_string(),
                "transcription".to_string(),
            ],
            default_model: Some("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string()),
            supports_reasoning: false,
            api_key_env: Some("TOGETHER_API_KEY".to_string()),
            api_key_env_aliases: vec!["TOGETHER_AI_API_KEY".to_string()],
        },
    );
    providers.insert(
        "togetherai".to_string(),
        ProviderConfig {
            id: "togetherai".to_string(),
            name: "TogetherAI".to_string(),
            base_url: "https://api.together.xyz/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "completion".to_string(),
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
                "image_generation".to_string(),
                "speech".to_string(),
                "transcription".to_string(),
            ],
            default_model: Some("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo".to_string()),
            supports_reasoning: false,
            api_key_env: Some("TOGETHER_API_KEY".to_string()),
            api_key_env_aliases: vec!["TOGETHER_AI_API_KEY".to_string()],
        },
    );

    // DeepInfra - OpenAI-compatible chat/completion/embedding surface with provider-owned image
    // routes
    providers.insert(
        "deepinfra".to_string(),
        ProviderConfig {
            id: "deepinfra".to_string(),
            name: "DeepInfra".to_string(),
            base_url: "https://api.deepinfra.com/v1/openai".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "completion".to_string(),
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
            ],
            default_model: Some("meta-llama/Llama-3.3-70B-Instruct".to_string()),
            supports_reasoning: false,
            api_key_env: Some("DEEPINFRA_API_KEY".to_string()),
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
            // Fireworks documents OpenAI-compatible `/chat/completions` and `/embeddings`.
            // It also documents a dedicated audio host (`https://audio.fireworks.ai/v1`)
            // for OpenAI-style `/audio/transcriptions`, so we enroll transcription only.
            // Note: `/responses` is documented but currently out-of-scope for Siumai's compat preset.
            capabilities: vec![
                "completion".to_string(),
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
                "transcription".to_string(),
            ],
            default_model: Some(fireworks_models::CHAT.to_string()),
            supports_reasoning: false,
            api_key_env: Some("FIREWORKS_API_KEY".to_string()),
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
            default_model: Some(perplexity_models::CHAT.to_string()),
            supports_reasoning: false,
            api_key_env: Some("PERPLEXITY_API_KEY".to_string()),
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
            capabilities: vec!["tools".to_string(), "transcription".to_string()],
            default_model: Some("llama-3.3-70b-versatile".to_string()),
            supports_reasoning: false,
            api_key_env: Some("GROQ_API_KEY".to_string()),
            api_key_env_aliases: Vec::new(),
        },
    );

    // xAI - Grok models (OpenAI-compatible)
    // Docs: https://x.ai/api
    providers.insert(
        "xai".to_string(),
        ProviderConfig {
            id: "xai".to_string(),
            name: "xAI".to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some("grok-2-1212".to_string()),
            supports_reasoning: false,
            api_key_env: Some("XAI_API_KEY".to_string()),
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
            default_model: Some(mistral_models::CHAT.to_string()),
            supports_reasoning: false,
            api_key_env: Some("MISTRAL_API_KEY".to_string()),
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

    // Moonshot AI - AI SDK package-aligned chat/language-model surface
    // Reference: repo-ref/ai/packages/moonshotai
    let moonshotai = ProviderConfig {
        id: "moonshotai".to_string(),
        name: "Moonshot AI".to_string(),
        base_url: "https://api.moonshot.ai/v1".to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec![
            "tools".to_string(),
            "vision".to_string(),
            "reasoning".to_string(),
        ],
        default_model: Some("kimi-k2-0905".to_string()),
        supports_reasoning: true,
        api_key_env: Some("MOONSHOT_API_KEY".to_string()),
        api_key_env_aliases: Vec::new(),
    };
    providers.insert("moonshotai".to_string(), moonshotai.clone());
    // Hidden compatibility alias kept during migration from the older pre-package id.
    providers.insert("moonshot".to_string(), moonshotai);

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

    // Alibaba Cloud / Qwen - OpenAI-compatible DashScope chat-completions surface.
    let qwen_capabilities = vec![
        "tools".to_string(),
        "vision".to_string(),
        "reasoning".to_string(),
    ];
    providers.insert(
        "alibaba".to_string(),
        ProviderConfig {
            id: "alibaba".to_string(),
            name: "Alibaba Cloud".to_string(),
            base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: qwen_capabilities.clone(),
            default_model: Some("qwen-plus".to_string()),
            supports_reasoning: true,
            api_key_env: Some("ALIBABA_API_KEY".to_string()),
            api_key_env_aliases: vec!["DASHSCOPE_API_KEY".to_string(), "QWEN_API_KEY".to_string()],
        },
    );
    // Historical local preset for the same model family, kept on the domestic DashScope endpoint.
    providers.insert(
        "qwen".to_string(),
        ProviderConfig {
            id: "qwen".to_string(),
            name: "Qwen".to_string(),
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: qwen_capabilities,
            default_model: Some("qwen-plus".to_string()),
            supports_reasoning: true,
            api_key_env: Some("ALIBABA_API_KEY".to_string()),
            api_key_env_aliases: vec!["DASHSCOPE_API_KEY".to_string()],
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

    // Google Vertex MaaS - OpenAI-compatible partner/open-model endpoint
    providers.insert(
        "vertex-maas".to_string(),
        ProviderConfig {
            id: "vertex-maas".to_string(),
            name: "Vertex MaaS".to_string(),
            // Real builds should override this with project/location derived URLs.
            base_url:
                "https://aiplatform.googleapis.com/v1/projects/PROJECT/locations/global/endpoints/openapi"
                    .to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
            ],
            default_model: Some("deepseek-ai/deepseek-v3.2-maas".to_string()),
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

static BUILTIN_PROVIDERS: LazyLock<HashMap<String, ProviderConfig>> =
    LazyLock::new(build_builtin_providers);

pub(crate) fn get_builtin_provider_map() -> &'static HashMap<String, ProviderConfig> {
    &BUILTIN_PROVIDERS
}

/// Get all built-in provider configurations
pub fn get_builtin_providers() -> HashMap<String, ProviderConfig> {
    get_builtin_provider_map().clone()
}

/// Build a generic OpenAI-compatible provider configuration.
///
/// This mirrors AI SDK's `createOpenAICompatible({ name, baseURL, ... })` path: callers provide a
/// provider name and endpoint, and Siumai uses the plain OpenAI-compatible adapter without applying
/// any built-in provider-specific transforms.
pub fn generic_provider_config(provider_id: &str, name: &str, base_url: &str) -> ProviderConfig {
    ProviderConfig {
        id: provider_id.to_string(),
        name: name.to_string(),
        base_url: base_url.to_string(),
        field_mappings: ProviderFieldMappings::default(),
        capabilities: vec![
            "tools".to_string(),
            "vision".to_string(),
            "embedding".to_string(),
            "image_generation".to_string(),
        ],
        default_model: None,
        supports_reasoning: false,
        api_key_env: None,
        api_key_env_aliases: Vec::new(),
    }
}

/// Get provider configuration by ID
pub fn get_provider_config(provider_id: &str) -> Option<ProviderConfig> {
    get_provider_config_ref(provider_id).cloned()
}

pub(crate) fn get_provider_config_ref(provider_id: &str) -> Option<&'static ProviderConfig> {
    get_builtin_provider_map().get(provider_id)
}

pub(crate) fn is_hidden_compat_alias(provider_id: &str) -> bool {
    matches!(provider_id, "together" | "moonshot")
}

/// List all available provider IDs
pub fn list_provider_ids() -> Vec<String> {
    get_builtin_provider_map()
        .keys()
        .filter(|provider_id| !is_hidden_compat_alias(provider_id))
        .cloned()
        .collect()
}

/// Check if a provider supports a specific capability
pub fn provider_supports_capability(provider_id: &str, capability: &str) -> bool {
    if let Some(config) = get_provider_config(provider_id) {
        ConfigurableAdapter::new(config)
            .capabilities()
            .supports(capability)
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_facade_keeps_family_defaults_split() {
        let source = include_str!("config.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap_or_default();
        let family_defaults = include_str!("config/family_defaults.rs");

        assert!(
            source.contains("mod family_defaults;"),
            "OpenAI-compatible config facade should keep family defaults in a dedicated module"
        );
        assert!(
            !source.contains("struct ProviderFamilyDefaults"),
            "OpenAI-compatible config facade should not own provider family defaults data"
        );
        assert!(
            family_defaults.contains("struct ProviderFamilyDefaults"),
            "provider family defaults should live in config/family_defaults.rs"
        );
    }

    #[test]
    fn test_builtin_providers() {
        let providers = get_builtin_providers();

        // Should have all major providers
        assert!(providers.contains_key("deepseek"));
        assert!(providers.contains_key("openrouter"));
        assert!(providers.contains_key("togetherai"));
        assert!(providers.contains_key("deepinfra"));
        assert!(providers.contains_key("vertex-maas"));
        assert!(providers.contains_key("fireworks"));
        assert!(providers.contains_key("moonshotai"));
        assert!(providers.contains_key("alibaba"));

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
        assert!(provider_supports_capability("siliconflow", "speech"));
        assert!(provider_supports_capability("siliconflow", "transcription"));
        assert!(provider_supports_capability("siliconflow", "audio"));
        assert!(provider_supports_capability("togetherai", "embedding"));
        assert!(provider_supports_capability("togetherai", "completion"));
        assert!(provider_supports_capability(
            "togetherai",
            "image_generation"
        ));
        assert!(provider_supports_capability("togetherai", "speech"));
        assert!(provider_supports_capability("togetherai", "transcription"));
        assert!(provider_supports_capability("togetherai", "audio"));
        assert!(provider_supports_capability("deepinfra", "embedding"));
        assert!(provider_supports_capability("deepinfra", "completion"));
        assert!(!provider_supports_capability(
            "deepinfra",
            "image_generation"
        ));
        assert!(provider_supports_capability("moonshotai", "reasoning"));
        assert!(!provider_supports_capability("moonshotai", "completion"));
        assert!(provider_supports_capability("alibaba", "reasoning"));
        assert!(provider_supports_capability("qwen", "reasoning"));
        assert!(provider_supports_capability("vertex-maas", "embedding"));
        assert!(!provider_supports_capability(
            "vertex-maas",
            "image_generation"
        ));
        assert!(provider_supports_capability("mistral", "embedding"));
        assert!(!provider_supports_capability("mistral", "completion"));
        assert!(!provider_supports_capability("perplexity", "completion"));
        assert!(provider_supports_capability("fireworks", "completion"));
        assert!(!provider_supports_capability("groq", "vision"));
        assert!(provider_supports_capability("groq", "transcription"));
        assert!(provider_supports_capability("groq", "audio"));
        assert!(!provider_supports_capability("openrouter", "speech"));
        assert!(!provider_supports_capability("openrouter", "transcription"));
        assert!(!provider_supports_capability("openrouter", "audio"));
        assert!(!provider_supports_capability("xai", "speech"));
        assert!(!provider_supports_capability("xai", "transcription"));
        assert!(!provider_supports_capability("xai", "audio"));
        assert!(!provider_supports_capability("fireworks", "speech"));
        assert!(provider_supports_capability("fireworks", "transcription"));
        assert!(provider_supports_capability("fireworks", "audio"));
    }

    #[test]
    fn test_provider_config_retrieval() {
        let config = get_provider_config("deepseek").unwrap();
        assert_eq!(config.id, "deepseek");
        assert_eq!(config.name, "DeepSeek");
        assert_eq!(config.base_url, "https://api.deepseek.com");
        assert_eq!(config.api_key_env.as_deref(), Some("DEEPSEEK_API_KEY"));

        let config = get_provider_config("deepinfra").unwrap();
        assert_eq!(config.id, "deepinfra");
        assert_eq!(config.name, "DeepInfra");
        assert_eq!(config.base_url, "https://api.deepinfra.com/v1/openai");
        assert_eq!(config.api_key_env.as_deref(), Some("DEEPINFRA_API_KEY"));

        let config = get_provider_config("togetherai").unwrap();
        assert_eq!(config.id, "togetherai");
        assert_eq!(config.api_key_env.as_deref(), Some("TOGETHER_API_KEY"));
        assert_eq!(
            config.api_key_env_aliases,
            vec!["TOGETHER_AI_API_KEY".to_string()]
        );

        let config = get_provider_config("moonshotai").unwrap();
        assert_eq!(config.id, "moonshotai");
        assert_eq!(config.base_url, "https://api.moonshot.ai/v1");
        assert_eq!(config.api_key_env.as_deref(), Some("MOONSHOT_API_KEY"));

        let config = get_provider_config("mistral").unwrap();
        assert_eq!(config.id, "mistral");
        assert_eq!(config.api_key_env.as_deref(), Some("MISTRAL_API_KEY"));

        let config = get_provider_config("groq").unwrap();
        assert_eq!(config.id, "groq");
        assert_eq!(config.base_url, "https://api.groq.com/openai/v1");
        assert_eq!(config.api_key_env.as_deref(), Some("GROQ_API_KEY"));

        let config = get_provider_config("xai").unwrap();
        assert_eq!(config.id, "xai");
        assert_eq!(config.api_key_env.as_deref(), Some("XAI_API_KEY"));

        let config = get_provider_config("perplexity").unwrap();
        assert_eq!(config.id, "perplexity");
        assert_eq!(config.api_key_env.as_deref(), Some("PERPLEXITY_API_KEY"));

        let config = get_provider_config("fireworks").unwrap();
        assert_eq!(config.id, "fireworks");
        assert_eq!(config.api_key_env.as_deref(), Some("FIREWORKS_API_KEY"));

        let config = get_provider_config("qwen").unwrap();
        assert_eq!(config.id, "qwen");
        assert_eq!(config.api_key_env.as_deref(), Some("ALIBABA_API_KEY"));
        assert_eq!(
            config.api_key_env_aliases,
            vec!["DASHSCOPE_API_KEY".to_string()]
        );

        let config = get_provider_config("alibaba").unwrap();
        assert_eq!(config.id, "alibaba");
        assert_eq!(
            config.base_url,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        );
        assert_eq!(config.api_key_env.as_deref(), Some("ALIBABA_API_KEY"));
        assert_eq!(
            config.api_key_env_aliases,
            vec!["DASHSCOPE_API_KEY".to_string(), "QWEN_API_KEY".to_string()]
        );

        assert!(get_provider_config("nonexistent").is_none());
    }

    #[test]
    fn test_moonshotai_canonical_id_hides_legacy_alias() {
        let provider_ids = list_provider_ids();
        assert!(
            provider_ids
                .iter()
                .any(|provider_id| provider_id == "moonshotai")
        );
        assert!(
            !provider_ids
                .iter()
                .any(|provider_id| provider_id == "moonshot")
        );

        let canonical = get_provider_config("moonshotai").expect("moonshotai provider config");
        let alias = get_provider_config("moonshot").expect("moonshot alias provider config");

        assert_eq!(canonical.id, "moonshotai");
        assert_eq!(alias.id, "moonshotai");
        assert_eq!(alias.base_url, canonical.base_url);
        assert_eq!(alias.default_model, canonical.default_model);
        assert!(provider_supports_capability("moonshotai", "reasoning"));
        assert!(!provider_supports_capability("moonshotai", "completion"));
    }

    #[test]
    fn test_completion_capable_hybrid_presets_keep_explicit_completion_metadata() {
        for provider_id in ["together", "togetherai", "deepinfra", "fireworks"] {
            let config = get_provider_config(provider_id)
                .unwrap_or_else(|| panic!("missing provider config for {provider_id}"));
            assert!(
                config.capabilities.iter().any(|cap| cap == "completion"),
                "expected {provider_id} compat preset metadata to advertise completion explicitly"
            );
        }
    }

    #[test]
    fn test_provider_family_defaults() {
        let openrouter =
            get_provider_family_defaults_ref("openrouter").expect("openrouter defaults");
        assert_eq!(openrouter.chat_model_override, Some("openai/gpt-4o"));
        assert_eq!(openrouter.embedding_model, Some("text-embedding-3-small"));

        let siliconflow =
            get_provider_family_defaults_ref("siliconflow").expect("siliconflow defaults");
        assert_eq!(siliconflow.embedding_model, Some("BAAI/bge-large-zh-v1.5"));
        assert_eq!(
            siliconflow.image_model,
            Some("stabilityai/stable-diffusion-3.5-large")
        );
        assert_eq!(siliconflow.rerank_model, Some("BAAI/bge-reranker-v2-m3"));
        assert_eq!(
            siliconflow.speech_model,
            Some("FunAudioLLM/CosyVoice2-0.5B")
        );
        assert_eq!(
            siliconflow.transcription_model,
            Some("FunAudioLLM/SenseVoiceSmall")
        );

        let togetherai =
            get_provider_family_defaults_ref("togetherai").expect("togetherai defaults");
        assert_eq!(
            togetherai.embedding_model,
            Some("togethercomputer/m2-bert-80M-8k-retrieval")
        );
        assert_eq!(
            togetherai.image_model,
            Some("black-forest-labs/FLUX.1-schnell")
        );
        assert_eq!(togetherai.speech_model, Some("cartesia/sonic-2"));
        assert_eq!(
            togetherai.transcription_model,
            Some("openai/whisper-large-v3")
        );

        let groq = get_provider_family_defaults_ref("groq").expect("groq defaults");
        assert_eq!(groq.transcription_model, Some(groq_models::TRANSCRIPTION));

        let deepinfra = get_provider_family_defaults_ref("deepinfra").expect("deepinfra defaults");
        assert_eq!(deepinfra.embedding_model, Some("BAAI/bge-base-en-v1.5"));
        assert_eq!(
            deepinfra.image_model,
            Some("black-forest-labs/FLUX-1-schnell")
        );
        assert_eq!(deepinfra.speech_model, None);
        assert_eq!(deepinfra.transcription_model, None);

        let fireworks = get_provider_family_defaults_ref("fireworks").expect("fireworks defaults");
        assert_eq!(
            fireworks.embedding_model,
            Some("nomic-ai/nomic-embed-text-v1.5")
        );
        assert_eq!(
            fireworks.image_model,
            Some("accounts/fireworks/models/flux-1-dev-fp8")
        );
        assert_eq!(fireworks.transcription_model, Some("whisper-v3"));
        assert_eq!(fireworks.speech_model, None);

        let mistral = get_provider_family_defaults_ref("mistral").expect("mistral defaults");
        assert_eq!(mistral.embedding_model, Some("mistral-embed"));
        assert_eq!(mistral.image_model, None);

        assert_eq!(
            get_default_embedding_model("infini"),
            Some("text-embedding-3-small")
        );
        assert_eq!(get_default_rerank_model("voyageai"), Some("rerank-2"));
        assert_eq!(get_default_rerank_model("jina"), Some("jina-reranker-m0"));
        assert_eq!(get_default_speech_model("openrouter"), None);
        assert_eq!(get_default_transcription_model("openrouter"), None);
    }
}
