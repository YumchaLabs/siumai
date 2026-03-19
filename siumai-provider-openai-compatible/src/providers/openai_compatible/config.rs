//! OpenAI-Compatible Providers Configuration
//!
//! This module contains centralized configuration for all OpenAI-compatible providers.
//! Inspired by Cherry Studio's provider configuration system.

use crate::providers::openai_compatible::ProviderAdapter;
use crate::standards::openai::compat::provider_registry::{
    ConfigurableAdapter, ProviderConfig, ProviderFieldMappings,
};
use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ProviderFamilyDefaults {
    pub chat_model_override: Option<&'static str>,
    pub embedding_model: Option<&'static str>,
    pub image_model: Option<&'static str>,
    pub rerank_model: Option<&'static str>,
    pub speech_model: Option<&'static str>,
    pub transcription_model: Option<&'static str>,
}

impl ProviderFamilyDefaults {
    pub const fn new() -> Self {
        Self {
            chat_model_override: None,
            embedding_model: None,
            image_model: None,
            rerank_model: None,
            speech_model: None,
            transcription_model: None,
        }
    }

    pub const fn with_chat_override(mut self, model: &'static str) -> Self {
        self.chat_model_override = Some(model);
        self
    }

    pub const fn with_embedding(mut self, model: &'static str) -> Self {
        self.embedding_model = Some(model);
        self
    }

    pub const fn with_image(mut self, model: &'static str) -> Self {
        self.image_model = Some(model);
        self
    }

    pub const fn with_rerank(mut self, model: &'static str) -> Self {
        self.rerank_model = Some(model);
        self
    }

    pub const fn with_speech(mut self, model: &'static str) -> Self {
        self.speech_model = Some(model);
        self
    }

    pub const fn with_transcription(mut self, model: &'static str) -> Self {
        self.transcription_model = Some(model);
        self
    }
}

fn build_builtin_provider_family_defaults() -> HashMap<&'static str, ProviderFamilyDefaults> {
    let mut defaults = HashMap::new();

    defaults.insert(
        "openrouter",
        ProviderFamilyDefaults::new()
            .with_chat_override("openai/gpt-4o")
            .with_embedding("text-embedding-3-small"),
    );
    defaults.insert(
        "deepseek",
        ProviderFamilyDefaults::new().with_embedding("deepseek-embedding"),
    );
    defaults.insert(
        "siliconflow",
        ProviderFamilyDefaults::new()
            .with_embedding("BAAI/bge-large-zh-v1.5")
            .with_image("stabilityai/stable-diffusion-3.5-large")
            .with_rerank("BAAI/bge-reranker-v2-m3")
            .with_speech("FunAudioLLM/CosyVoice2-0.5B")
            .with_transcription("FunAudioLLM/SenseVoiceSmall"),
    );
    defaults.insert(
        "together",
        ProviderFamilyDefaults::new()
            .with_embedding("togethercomputer/m2-bert-80M-8k-retrieval")
            .with_image("black-forest-labs/FLUX.1-schnell")
            .with_speech("cartesia/sonic-2")
            .with_transcription("openai/whisper-large-v3"),
    );
    defaults.insert(
        "fireworks",
        ProviderFamilyDefaults::new()
            .with_embedding("nomic-ai/nomic-embed-text-v1.5")
            .with_transcription("whisper-v3"),
    );
    defaults.insert(
        "jina",
        ProviderFamilyDefaults::new()
            .with_embedding("jina-embeddings-v2-base-en")
            .with_rerank("jina-reranker-m0"),
    );
    defaults.insert(
        "voyageai",
        ProviderFamilyDefaults::new()
            .with_embedding("voyage-3")
            .with_rerank("rerank-2"),
    );
    defaults.insert(
        "infini",
        ProviderFamilyDefaults::new().with_embedding("text-embedding-3-small"),
    );

    defaults
}

static BUILTIN_PROVIDER_FAMILY_DEFAULTS: LazyLock<HashMap<&'static str, ProviderFamilyDefaults>> =
    LazyLock::new(build_builtin_provider_family_defaults);

pub(crate) fn get_builtin_provider_family_defaults_map()
-> &'static HashMap<&'static str, ProviderFamilyDefaults> {
    &BUILTIN_PROVIDER_FAMILY_DEFAULTS
}

pub(crate) fn get_provider_family_defaults_ref(
    provider_id: &str,
) -> Option<&'static ProviderFamilyDefaults> {
    get_builtin_provider_family_defaults_map().get(provider_id)
}

pub(crate) fn get_default_embedding_model(provider_id: &str) -> Option<&'static str> {
    get_provider_family_defaults_ref(provider_id).and_then(|defaults| defaults.embedding_model)
}

pub(crate) fn get_default_image_model(provider_id: &str) -> Option<&'static str> {
    get_provider_family_defaults_ref(provider_id).and_then(|defaults| defaults.image_model)
}

pub(crate) fn get_default_rerank_model(provider_id: &str) -> Option<&'static str> {
    get_provider_family_defaults_ref(provider_id).and_then(|defaults| defaults.rerank_model)
}

pub(crate) fn get_default_speech_model(provider_id: &str) -> Option<&'static str> {
    get_provider_family_defaults_ref(provider_id).and_then(|defaults| defaults.speech_model)
}

pub(crate) fn get_default_transcription_model(provider_id: &str) -> Option<&'static str> {
    get_provider_family_defaults_ref(provider_id).and_then(|defaults| defaults.transcription_model)
}

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
            // Together documents OpenAI-style `/embeddings` and `/images/generations` endpoints.
            capabilities: vec![
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
            // Fireworks documents OpenAI-compatible `/chat/completions` and `/embeddings`.
            // It also documents a dedicated audio host (`https://audio.fireworks.ai/v1`)
            // for OpenAI-style `/audio/transcriptions`, so we enroll transcription only.
            // Note: `/responses` is documented but currently out-of-scope for Siumai's compat preset.
            capabilities: vec![
                "tools".to_string(),
                "vision".to_string(),
                "embedding".to_string(),
                "transcription".to_string(),
            ],
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
            default_model: Some("llama-3.3-70b-versatile".to_string()),
            supports_reasoning: false,
            api_key_env: None,
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

static BUILTIN_PROVIDERS: LazyLock<HashMap<String, ProviderConfig>> =
    LazyLock::new(build_builtin_providers);

pub(crate) fn get_builtin_provider_map() -> &'static HashMap<String, ProviderConfig> {
    &BUILTIN_PROVIDERS
}

/// Get all built-in provider configurations
pub fn get_builtin_providers() -> HashMap<String, ProviderConfig> {
    get_builtin_provider_map().clone()
}

/// Get provider configuration by ID
pub fn get_provider_config(provider_id: &str) -> Option<ProviderConfig> {
    get_provider_config_ref(provider_id).cloned()
}

pub(crate) fn get_provider_config_ref(provider_id: &str) -> Option<&'static ProviderConfig> {
    get_builtin_provider_map().get(provider_id)
}

/// List all available provider IDs
pub fn list_provider_ids() -> Vec<String> {
    get_builtin_provider_map().keys().cloned().collect()
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
        assert!(provider_supports_capability("siliconflow", "speech"));
        assert!(provider_supports_capability("siliconflow", "transcription"));
        assert!(provider_supports_capability("siliconflow", "audio"));
        assert!(provider_supports_capability("together", "embedding"));
        assert!(provider_supports_capability("together", "image_generation"));
        assert!(provider_supports_capability("together", "speech"));
        assert!(provider_supports_capability("together", "transcription"));
        assert!(provider_supports_capability("together", "audio"));
        assert!(provider_supports_capability("mistral", "embedding"));
        assert!(!provider_supports_capability("groq", "vision"));
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

        assert!(get_provider_config("nonexistent").is_none());
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

        let together = get_provider_family_defaults_ref("together").expect("together defaults");
        assert_eq!(
            together.embedding_model,
            Some("togethercomputer/m2-bert-80M-8k-retrieval")
        );
        assert_eq!(
            together.image_model,
            Some("black-forest-labs/FLUX.1-schnell")
        );
        assert_eq!(together.speech_model, Some("cartesia/sonic-2"));
        assert_eq!(
            together.transcription_model,
            Some("openai/whisper-large-v3")
        );

        let fireworks = get_provider_family_defaults_ref("fireworks").expect("fireworks defaults");
        assert_eq!(
            fireworks.embedding_model,
            Some("nomic-ai/nomic-embed-text-v1.5")
        );
        assert_eq!(fireworks.transcription_model, Some("whisper-v3"));
        assert_eq!(fireworks.speech_model, None);

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
