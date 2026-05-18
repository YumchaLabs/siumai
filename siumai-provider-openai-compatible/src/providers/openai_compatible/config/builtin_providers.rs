use crate::providers::openai_compatible::fireworks as fireworks_models;
use crate::providers::openai_compatible::google_vertex_xai as google_vertex_xai_models;
use crate::providers::openai_compatible::mistral as mistral_models;
use crate::providers::openai_compatible::perplexity as perplexity_models;
use crate::standards::openai::compat::provider_registry::{ProviderConfig, ProviderFieldMappings};
use std::collections::HashMap;
use std::sync::LazyLock;

/// Build all built-in OpenAI-compatible provider configurations.
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
                "completion".to_string(),
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

    // Google Vertex xAI - Grok partner models via Vertex OpenAI-compatible endpoint.
    providers.insert(
        "google-vertex-xai".to_string(),
        ProviderConfig {
            id: "google-vertex-xai".to_string(),
            name: "Google Vertex xAI".to_string(),
            // Real builds should override this with project/location derived URLs.
            base_url:
                "https://aiplatform.googleapis.com/v1/projects/PROJECT/locations/global/endpoints/openapi"
                    .to_string(),
            field_mappings: ProviderFieldMappings::default(),
            capabilities: vec!["tools".to_string(), "vision".to_string()],
            default_model: Some(google_vertex_xai_models::CHAT.to_string()),
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
