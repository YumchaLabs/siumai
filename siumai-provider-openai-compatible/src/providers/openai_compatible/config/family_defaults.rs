use crate::providers::openai_compatible::fireworks as fireworks_models;
use crate::providers::openai_compatible::groq as groq_models;
use crate::providers::openai_compatible::mistral as mistral_models;
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
    pub include_usage: Option<bool>,
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
            include_usage: None,
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

    pub const fn with_include_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = Some(include_usage);
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
        ProviderFamilyDefaults::new()
            .with_embedding("deepseek-embedding")
            .with_include_usage(true),
    );
    defaults.insert(
        "siliconflow",
        ProviderFamilyDefaults::new()
            .with_embedding("BAAI/bge-large-zh-v1.5")
            .with_image("stabilityai/stable-diffusion-3.5-large")
            .with_rerank("BAAI/bge-reranker-v2-m3")
            .with_speech("FunAudioLLM/CosyVoice2-0.5B")
            .with_transcription("FunAudioLLM/SenseVoiceSmall")
            .with_include_usage(true),
    );
    defaults.insert(
        "xai",
        ProviderFamilyDefaults::new().with_include_usage(true),
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
        "togetherai",
        ProviderFamilyDefaults::new()
            .with_embedding("togethercomputer/m2-bert-80M-8k-retrieval")
            .with_image("black-forest-labs/FLUX.1-schnell")
            .with_speech("cartesia/sonic-2")
            .with_transcription("openai/whisper-large-v3"),
    );
    defaults.insert(
        "deepinfra",
        ProviderFamilyDefaults::new()
            .with_embedding("BAAI/bge-base-en-v1.5")
            .with_image("black-forest-labs/FLUX-1-schnell"),
    );
    defaults.insert(
        "fireworks",
        ProviderFamilyDefaults::new()
            .with_embedding(fireworks_models::EMBEDDING)
            .with_image(fireworks_models::IMAGE)
            .with_transcription("whisper-v3"),
    );
    defaults.insert(
        "groq",
        ProviderFamilyDefaults::new().with_transcription(groq_models::TRANSCRIPTION),
    );
    defaults.insert(
        "mistral",
        ProviderFamilyDefaults::new().with_embedding(mistral_models::EMBEDDING),
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
    defaults.insert(
        "moonshotai",
        ProviderFamilyDefaults::new().with_include_usage(true),
    );
    defaults.insert(
        "alibaba",
        ProviderFamilyDefaults::new().with_include_usage(true),
    );
    defaults.insert(
        "qwen",
        ProviderFamilyDefaults::new().with_include_usage(true),
    );
    defaults.insert(
        "google-vertex-xai",
        ProviderFamilyDefaults::new().with_include_usage(true),
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

pub(crate) fn default_include_usage(provider_id: &str) -> Option<bool> {
    get_provider_family_defaults_ref(provider_id).and_then(|defaults| defaults.include_usage)
}
