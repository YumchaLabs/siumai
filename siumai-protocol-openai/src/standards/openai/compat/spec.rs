use crate::core::{
    AudioTransformer as AudioTransformerBundle, ChatTransformers, EmbeddingTransformers,
    ImageTransformers, ProviderContext, ProviderSpec, RerankTransformers,
};
use crate::error::LlmError;
use crate::standards::openai::compat::adapter::OpenAiCompatibleRequestSettings;
use crate::traits::ProviderCapabilities;
use crate::types::{
    ChatRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest, RerankRequest,
    Warning,
};
use reqwest::header::HeaderMap;
use std::sync::Arc;

fn rename_field(obj: &mut serde_json::Map<String, serde_json::Value>, from: &str, to: &str) {
    if let Some(v) = obj.remove(from) {
        obj.entry(to.to_string()).or_insert(v);
    }
}

fn normalize_xai_search_parameters(v: &mut serde_json::Value) {
    let Some(obj) = v.as_object_mut() else {
        return;
    };

    rename_field(obj, "returnCitations", "return_citations");
    rename_field(obj, "maxSearchResults", "max_search_results");
    rename_field(obj, "fromDate", "from_date");
    rename_field(obj, "toDate", "to_date");

    if let Some(arr) = obj.get_mut("sources").and_then(|v| v.as_array_mut()) {
        for src in arr {
            let Some(src_obj) = src.as_object_mut() else {
                continue;
            };

            rename_field(src_obj, "allowedWebsites", "allowed_websites");
            rename_field(src_obj, "excludedWebsites", "excluded_websites");
            rename_field(src_obj, "safeSearch", "safe_search");

            rename_field(src_obj, "excludedXHandles", "excluded_x_handles");
            rename_field(src_obj, "includedXHandles", "included_x_handles");
            rename_field(src_obj, "postFavoriteCount", "post_favorite_count");
            rename_field(src_obj, "postViewCount", "post_view_count");
            rename_field(src_obj, "xHandles", "x_handles");

            if src_obj.get("included_x_handles").is_none() {
                if let Some(value) = src_obj.remove("x_handles") {
                    src_obj.insert("included_x_handles".to_string(), value);
                }
            } else {
                src_obj.remove("x_handles");
            }
        }
    }
}

fn normalize_deepseek_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
    rename_field(obj, "enableReasoning", "enable_reasoning");
    rename_field(obj, "reasoningBudget", "reasoning_budget");
}

fn string_option_by_aliases(
    obj: &serde_json::Map<String, serde_json::Value>,
    aliases: &[&str],
) -> Option<String> {
    aliases
        .iter()
        .find_map(|alias| obj.get(*alias))
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

fn bool_option_by_aliases(
    obj: &serde_json::Map<String, serde_json::Value>,
    aliases: &[&str],
) -> Option<bool> {
    aliases
        .iter()
        .find_map(|alias| obj.get(*alias))
        .and_then(|value| value.as_bool())
}

#[derive(Debug, Clone, Default)]
struct CompatChatOptions {
    user: Option<String>,
    reasoning_effort: Option<String>,
    verbosity: Option<String>,
    strict_json_schema: Option<bool>,
}

fn compat_chat_options(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> CompatChatOptions {
    let mut out = CompatChatOptions::default();

    for options in [
        map.get_object("openai-compatible"),
        map.get_object("openaiCompatible"),
        map.get_object(provider_id),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(user) = string_option_by_aliases(options, &["user"]) {
            out.user = Some(user);
        }
        if let Some(reasoning_effort) =
            string_option_by_aliases(options, &["reasoningEffort", "reasoning_effort"])
        {
            out.reasoning_effort = Some(reasoning_effort);
        }
        if let Some(verbosity) =
            string_option_by_aliases(options, &["textVerbosity", "text_verbosity"])
        {
            out.verbosity = Some(verbosity);
        }
        if let Some(strict_json_schema) =
            bool_option_by_aliases(options, &["strictJsonSchema", "strict_json_schema"])
        {
            out.strict_json_schema = Some(strict_json_schema);
        }
    }

    out
}

fn remove_known_compat_chat_option_keys(obj: &mut serde_json::Map<String, serde_json::Value>) {
    for key in [
        "user",
        "reasoningEffort",
        "reasoning_effort",
        "textVerbosity",
        "text_verbosity",
        "strictJsonSchema",
        "strict_json_schema",
    ] {
        obj.remove(key);
    }
}

fn provider_options_key(provider_id: &str) -> String {
    provider_id
        .split('.')
        .next()
        .unwrap_or(provider_id)
        .trim()
        .to_ascii_lowercase()
}

fn to_camel_case(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut uppercase_next = false;

    for ch in value.chars() {
        if matches!(ch, '-' | '_') {
            uppercase_next = true;
            continue;
        }

        if uppercase_next {
            out.extend(ch.to_uppercase());
            uppercase_next = false;
        } else {
            out.push(ch);
        }
    }

    out
}

fn compat_image_provider_options(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let provider_key = provider_options_key(provider_id);
    let provider_camel = to_camel_case(&provider_key);
    let mut merged = serde_json::Map::new();

    for options in [
        map.get_object("openai-compatible"),
        map.get_object("openaiCompatible"),
        map.get_object(&provider_key),
        (provider_camel != provider_key)
            .then(|| map.get_object(&provider_camel))
            .flatten(),
    ]
    .into_iter()
    .flatten()
    {
        for (key, value) in options {
            merged.insert(key.clone(), value.clone());
        }
    }

    (!merged.is_empty()).then_some(merged)
}

fn normalize_provider_options(
    provider_id: &str,
    value: Option<&serde_json::Value>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    fn take_any(
        obj: &mut serde_json::Map<String, serde_json::Value>,
        keys: &[&str],
    ) -> Option<serde_json::Value> {
        for key in keys {
            if let Some(value) = obj.remove(*key) {
                return Some(value);
            }
        }
        None
    }

    let mut obj = value?.as_object()?.clone();

    if provider_id == "xai" {
        rename_field(&mut obj, "reasoningEffort", "reasoning_effort");
        rename_field(&mut obj, "reasoningSummary", "reasoning_summary");
        rename_field(&mut obj, "searchParameters", "search_parameters");
        rename_field(&mut obj, "topLogprobs", "top_logprobs");
        rename_field(&mut obj, "previousResponseId", "previous_response_id");
        if let Some(v) = obj.get_mut("search_parameters") {
            normalize_xai_search_parameters(v);
        }
        if obj.get("top_logprobs").is_some() {
            obj.insert("logprobs".to_string(), serde_json::json!(true));
        }
    } else if provider_id == "deepseek" {
        normalize_deepseek_options(&mut obj);
    } else if provider_id == "groq" {
        if let Some(value) = take_any(&mut obj, &["max_completion_tokens", "max_tokens"]) {
            obj.entry("max_tokens".to_string()).or_insert(value);
        }
    }

    Some(obj)
}

fn normalize_chat_passthrough_provider_options(
    provider_id: &str,
    value: Option<&serde_json::Value>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut obj = normalize_provider_options(provider_id, value)?;
    remove_known_compat_chat_option_keys(&mut obj);
    Some(obj)
}

fn merge_provider_options_into_body(
    provider_id: &str,
    provider_options: &Option<serde_json::Map<String, serde_json::Value>>,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    let Some(obj) = provider_options.as_ref() else {
        return;
    };

    for (k, v) in obj {
        if matches!(k.as_str(), "response_format" | "tool_choice") && body_obj.contains_key(k) {
            continue;
        }
        body_obj.insert(k.clone(), v.clone());
    }

    if provider_id == "xai" {
        body_obj.remove("stop");
        body_obj.remove("stream_options");
        body_obj.remove("reasoning_summary");
        body_obj.remove("previous_response_id");
        body_obj.remove("include");
        body_obj.remove("store");
    }
}

fn apply_compat_chat_options(
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
    compat_options: &CompatChatOptions,
) {
    if let Some(user) = compat_options.user.as_ref() {
        body_obj.insert("user".to_string(), serde_json::json!(user));
    }

    if let Some(reasoning_effort) = compat_options.reasoning_effort.as_ref() {
        body_obj.insert(
            "reasoning_effort".to_string(),
            serde_json::json!(reasoning_effort),
        );
    }

    if let Some(verbosity) = compat_options.verbosity.as_ref() {
        body_obj.insert("verbosity".to_string(), serde_json::json!(verbosity));
    }

    if let Some(strict_json_schema) = compat_options.strict_json_schema
        && let Some(response_format) = body_obj.get_mut("response_format")
        && let Some(response_format_obj) = response_format.as_object_mut()
        && response_format_obj
            .get("type")
            .and_then(|value| value.as_str())
            == Some("json_schema")
        && let Some(json_schema_obj) = response_format_obj
            .get_mut("json_schema")
            .and_then(|value| value.as_object_mut())
    {
        json_schema_obj.insert(
            "strict".to_string(),
            serde_json::Value::Bool(strict_json_schema),
        );
    }
}

fn apply_chat_request_settings(
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
    settings: &OpenAiCompatibleRequestSettings,
    supports_stream_usage_hints: bool,
    request_uses_structured_outputs: bool,
) {
    let stream = body_obj
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    if stream {
        if settings.include_usage == Some(true) && supports_stream_usage_hints {
            body_obj.insert(
                "stream_options".to_string(),
                serde_json::json!({ "include_usage": true }),
            );
        } else {
            body_obj.remove("stream_options");
        }
    }

    if settings.supports_structured_outputs != Some(true) && request_uses_structured_outputs {
        body_obj.insert(
            "response_format".to_string(),
            serde_json::json!({ "type": "json_object" }),
        );
    }
}

fn provider_options_map_merge_hook(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> Option<crate::execution::executors::BeforeSendHook> {
    let provider_id = provider_id.to_string();
    let provider_options = normalize_provider_options(&provider_id, map.get(&provider_id));
    if provider_options.is_none() {
        return None;
    }
    let hook = move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
        let mut out = body.clone();
        if let Some(body_obj) = out.as_object_mut() {
            merge_provider_options_into_body(&provider_id, &provider_options, body_obj);
        }
        Ok(out)
    };
    Some(Arc::new(hook))
}

fn chat_request_settings_hook(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
    settings: &OpenAiCompatibleRequestSettings,
    supports_stream_usage_hints: bool,
    request_uses_structured_outputs: bool,
) -> crate::execution::executors::BeforeSendHook {
    let provider_id = provider_id.to_string();
    let compat_options = compat_chat_options(&provider_id, map);
    let provider_options =
        normalize_chat_passthrough_provider_options(&provider_id, map.get(&provider_id));
    let settings = settings.clone();
    Arc::new(
        move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            if let Some(body_obj) = out.as_object_mut() {
                apply_compat_chat_options(body_obj, &compat_options);
                merge_provider_options_into_body(&provider_id, &provider_options, body_obj);
                apply_chat_request_settings(
                    body_obj,
                    &settings,
                    supports_stream_usage_hints,
                    request_uses_structured_outputs,
                );
            }

            if let Some(transformer) = settings.request_body_transformer.as_ref() {
                let model = out
                    .get("model")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string();
                transformer.transform_request_body(
                    &mut out,
                    &model,
                    super::types::RequestType::Chat,
                )?;
            }

            Ok(out)
        },
    )
}

fn apply_url_settings(url: String, settings: &OpenAiCompatibleRequestSettings) -> String {
    crate::utils::url::with_query_params(&url, &settings.query_params)
}

/// OpenAI-Compatible ProviderSpec implementation with an injected adapter.
///
/// This is used by OpenAI-compatible clients to avoid runtime global registry lookups.
#[derive(Clone)]
pub struct OpenAiCompatibleSpecWithAdapter {
    adapter: Arc<dyn super::adapter::ProviderAdapter>,
    request_settings: OpenAiCompatibleRequestSettings,
}

impl OpenAiCompatibleSpecWithAdapter {
    pub fn new(adapter: Arc<dyn super::adapter::ProviderAdapter>) -> Self {
        Self::with_settings(adapter, OpenAiCompatibleRequestSettings::default())
    }

    pub fn with_settings(
        adapter: Arc<dyn super::adapter::ProviderAdapter>,
        request_settings: OpenAiCompatibleRequestSettings,
    ) -> Self {
        Self {
            adapter,
            request_settings,
        }
    }
}

impl ProviderSpec for OpenAiCompatibleSpecWithAdapter {
    fn id(&self) -> &'static str {
        "openai_compatible"
    }

    fn capabilities(&self) -> ProviderCapabilities {
        // Prefer adapter-declared capabilities.
        self.adapter.capabilities()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        self.chat_spec().build_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        _headers: &HeaderMap,
    ) -> Option<LlmError> {
        let provider_id = self.adapter.provider_id();
        crate::standards::openai::errors::classify_openai_compatible_http_error(
            provider_id.as_ref(),
            status,
            body_text,
        )
    }

    fn chat_url(&self, stream: bool, req: &ChatRequest, ctx: &ProviderContext) -> String {
        apply_url_settings(
            self.chat_spec().chat_url(stream, req, ctx),
            &self.request_settings,
        )
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> ChatTransformers {
        self.chat_spec().choose_chat_transformers(req, ctx)
    }

    fn chat_before_send(
        &self,
        req: &crate::types::ChatRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        Some(chat_request_settings_hook(
            &ctx.provider_id,
            &req.provider_options_map,
            &self.request_settings,
            self.adapter.supports_stream_usage_hints(),
            req.response_format.is_some(),
        ))
    }

    fn choose_embedding_transformers(
        &self,
        req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> EmbeddingTransformers {
        self.embedding_spec()
            .choose_embedding_transformers(req, ctx)
    }

    fn embedding_before_send(
        &self,
        req: &crate::types::EmbeddingRequest,
        ctx: &ProviderContext,
    ) -> Option<crate::execution::executors::BeforeSendHook> {
        provider_options_map_merge_hook(&ctx.provider_id, &req.provider_options_map)
    }

    fn embedding_url(&self, req: &crate::types::EmbeddingRequest, ctx: &ProviderContext) -> String {
        apply_url_settings(
            self.embedding_spec().embedding_url(req, ctx),
            &self.request_settings,
        )
    }

    fn choose_image_transformers(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> ImageTransformers {
        self.image_spec().choose_image_transformers(req, ctx)
    }

    fn image_url(
        &self,
        req: &crate::types::ImageGenerationRequest,
        ctx: &ProviderContext,
    ) -> String {
        apply_url_settings(
            self.image_spec().image_url(req, ctx),
            &self.request_settings,
        )
    }

    fn image_warnings(
        &self,
        req: &ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        let mut warnings = Vec::new();

        if req.aspect_ratio.is_some() {
            warnings.push(Warning::unsupported(
                "aspectRatio",
                Some("This model does not support aspect ratio. Use `size` instead."),
            ));
        }

        if req.seed.is_some() {
            warnings.push(Warning::unsupported("seed", Option::<String>::None));
        }

        (!warnings.is_empty()).then_some(warnings)
    }

    fn image_edit_url(&self, req: &ImageEditRequest, ctx: &ProviderContext) -> String {
        apply_url_settings(
            self.image_spec().image_edit_url(req, ctx),
            &self.request_settings,
        )
    }

    fn image_edit_warnings(
        &self,
        req: &ImageEditRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        let mut warnings = Vec::new();

        if req.aspect_ratio.is_some() {
            warnings.push(Warning::unsupported(
                "aspectRatio",
                Some("This model does not support aspect ratio. Use `size` instead."),
            ));
        }

        if req.seed.is_some() {
            warnings.push(Warning::unsupported("seed", Option::<String>::None));
        }

        (!warnings.is_empty()).then_some(warnings)
    }

    fn image_variation_url(&self, req: &ImageVariationRequest, ctx: &ProviderContext) -> String {
        apply_url_settings(
            self.image_spec().image_variation_url(req, ctx),
            &self.request_settings,
        )
    }

    fn image_variation_warnings(
        &self,
        req: &ImageVariationRequest,
        _ctx: &ProviderContext,
    ) -> Option<Vec<Warning>> {
        let mut warnings = Vec::new();

        if req.aspect_ratio.is_some() {
            warnings.push(Warning::unsupported(
                "aspectRatio",
                Some("This model does not support aspect ratio. Use `size` instead."),
            ));
        }

        if req.seed.is_some() {
            warnings.push(Warning::unsupported("seed", Option::<String>::None));
        }

        (!warnings.is_empty()).then_some(warnings)
    }

    fn choose_audio_transformer(&self, ctx: &ProviderContext) -> AudioTransformerBundle {
        AudioTransformerBundle {
            transformer: Arc::new(
                crate::standards::openai::audio::OpenAiAudioTransformerWithProviderId::new(
                    ctx.provider_id.clone(),
                ),
            ),
        }
    }

    fn audio_base_url(&self, ctx: &ProviderContext) -> String {
        let ctx_base = ctx.base_url.trim_end_matches('/');
        let adapter_base = self.adapter.base_url().trim_end_matches('/');

        if ctx_base != adapter_base {
            return apply_url_settings(ctx_base.to_string(), &self.request_settings);
        }

        apply_url_settings(
            self.adapter
                .audio_base_url()
                .unwrap_or(adapter_base)
                .trim_end_matches('/')
                .to_string(),
            &self.request_settings,
        )
    }

    fn rerank_url(&self, req: &RerankRequest, ctx: &ProviderContext) -> String {
        apply_url_settings(
            self.rerank_spec().rerank_url(req, ctx),
            &self.request_settings,
        )
    }

    fn models_url(&self, ctx: &ProviderContext) -> String {
        apply_url_settings(
            format!("{}/models", ctx.base_url.trim_end_matches('/')),
            &self.request_settings,
        )
    }

    fn model_url(&self, model_id: &str, ctx: &ProviderContext) -> String {
        apply_url_settings(
            format!("{}/models/{}", ctx.base_url.trim_end_matches('/'), model_id),
            &self.request_settings,
        )
    }

    fn choose_rerank_transformers(
        &self,
        req: &RerankRequest,
        ctx: &ProviderContext,
    ) -> RerankTransformers {
        self.rerank_spec().choose_rerank_transformers(req, ctx)
    }
}

impl OpenAiCompatibleSpecWithAdapter {
    fn chat_spec(&self) -> crate::standards::openai::chat::OpenAiChatSpec {
        #[derive(Debug)]
        struct CompatToOpenAiChatAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            chat_endpoint: String,
        }

        impl CompatToOpenAiChatAdapter {
            fn new(adapter: Arc<dyn super::adapter::ProviderAdapter>) -> Self {
                let path = adapter.route_for(super::types::RequestType::Chat);
                let endpoint = format!("/{}", path.trim_start_matches('/'));
                Self {
                    adapter,
                    chat_endpoint: endpoint,
                }
            }
        }

        impl crate::standards::openai::chat::OpenAiChatAdapter for CompatToOpenAiChatAdapter {
            fn build_headers(
                &self,
                api_key: &str,
                base_headers: &mut reqwest::header::HeaderMap,
            ) -> Result<(), LlmError> {
                if api_key.is_empty() {
                    return Err(LlmError::MissingApiKey(
                        "OpenAI-Compatible API key not provided".into(),
                    ));
                }
                let _ = base_headers;
                Ok(())
            }

            fn transform_request(
                &self,
                req: &ChatRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.adapter.transform_request_params(
                    body,
                    &req.common_params.model,
                    super::types::RequestType::Chat,
                )
            }

            fn chat_endpoint(&self) -> &str {
                &self.chat_endpoint
            }
        }

        crate::standards::openai::chat::OpenAiChatStandard::with_adapters(
            Arc::new(CompatToOpenAiChatAdapter::new(self.adapter.clone())),
            self.adapter.clone(),
        )
        .create_spec("openai_compatible")
    }

    fn embedding_spec(&self) -> crate::standards::openai::embedding::OpenAiEmbeddingSpec {
        #[derive(Clone)]
        struct CompatEmbeddingAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            embedding_endpoint: String,
        }

        impl crate::standards::openai::embedding::OpenAiEmbeddingAdapter for CompatEmbeddingAdapter {
            fn transform_request(
                &self,
                req: &crate::types::EmbeddingRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.adapter.transform_request_params(
                    body,
                    req.model.as_deref().unwrap_or(""),
                    super::types::RequestType::Embedding,
                )
            }

            fn embedding_endpoint(&self) -> &str {
                &self.embedding_endpoint
            }
        }

        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::Embedding)
                .trim_start_matches('/')
        );

        crate::standards::openai::embedding::OpenAiEmbeddingStandard::with_adapter(Arc::new(
            CompatEmbeddingAdapter {
                adapter: self.adapter.clone(),
                embedding_endpoint: endpoint,
            },
        ))
        .create_spec("openai_compatible")
    }

    fn image_spec(&self) -> crate::standards::openai::image::OpenAiImageSpec {
        #[derive(Clone)]
        struct CompatImageAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            generation_endpoint: String,
        }

        impl crate::standards::openai::image::OpenAiImageAdapter for CompatImageAdapter {
            fn transform_generation_request(
                &self,
                _req: &crate::types::ImageGenerationRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                let model_s = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                self.adapter.transform_request_params(
                    body,
                    &model_s,
                    super::types::RequestType::ImageGeneration,
                )
            }

            fn generation_endpoint(&self) -> &str {
                &self.generation_endpoint
            }

            fn generation_provider_options(
                &self,
                req: &crate::types::ImageGenerationRequest,
            ) -> Option<serde_json::Map<String, serde_json::Value>> {
                compat_image_provider_options(
                    self.adapter.provider_id().as_ref(),
                    &req.provider_options_map,
                )
            }

            fn edit_provider_options(
                &self,
                req: &crate::types::ImageEditRequest,
            ) -> Option<serde_json::Map<String, serde_json::Value>> {
                compat_image_provider_options(
                    self.adapter.provider_id().as_ref(),
                    &req.provider_options_map,
                )
            }

            fn variation_provider_options(
                &self,
                req: &crate::types::ImageVariationRequest,
            ) -> Option<serde_json::Map<String, serde_json::Value>> {
                compat_image_provider_options(
                    self.adapter.provider_id().as_ref(),
                    &req.provider_options_map,
                )
            }
        }

        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::ImageGeneration)
                .trim_start_matches('/')
        );

        crate::standards::openai::image::OpenAiImageStandard::with_adapter(Arc::new(
            CompatImageAdapter {
                adapter: self.adapter.clone(),
                generation_endpoint: endpoint,
            },
        ))
        .create_spec("openai_compatible")
    }

    fn rerank_spec(&self) -> crate::standards::openai::rerank::OpenAiRerankSpec {
        #[derive(Clone)]
        struct CompatRerankAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            rerank_endpoint: String,
        }

        impl crate::standards::openai::rerank::OpenAiRerankAdapter for CompatRerankAdapter {
            fn transform_request(
                &self,
                req: &RerankRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                self.adapter.transform_request_params(
                    body,
                    &req.model,
                    super::types::RequestType::Rerank,
                )
            }

            fn rerank_endpoint(&self) -> &str {
                &self.rerank_endpoint
            }
        }

        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::Rerank)
                .trim_start_matches('/')
        );

        crate::standards::openai::rerank::OpenAiRerankStandard::with_adapter(Arc::new(
            CompatRerankAdapter {
                adapter: self.adapter.clone(),
                rerank_endpoint: endpoint,
            },
        ))
        .create_spec("openai_compatible")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standards::openai::compat::provider_registry::{
        ConfigurableAdapter, ProviderConfig,
    };

    fn compat_image_spec_and_ctx() -> (OpenAiCompatibleSpecWithAdapter, ProviderContext) {
        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["image_generation".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        (spec, ctx)
    }

    #[test]
    fn openai_compatible_audio_transformer_uses_openai_audio_endpoints() {
        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "compat-audio".to_string(),
                name: "Compat Audio".to_string(),
                base_url: "https://api.compat.example/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["speech".into(), "transcription".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "compat-audio".to_string(),
            "https://api.compat.example/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let transformer = spec.choose_audio_transformer(&ctx).transformer;

        assert!(spec.capabilities().supports("audio"));
        assert!(spec.capabilities().supports("speech"));
        assert!(spec.capabilities().supports("transcription"));
        assert_eq!(transformer.provider_id(), "compat-audio");
        assert_eq!(transformer.tts_endpoint(), "/audio/speech");
        assert_eq!(transformer.stt_endpoint(), "/audio/transcriptions");
    }

    #[test]
    fn openai_compatible_audio_uses_provider_audio_base_by_default() {
        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "fireworks".to_string(),
                name: "Fireworks AI".to_string(),
                base_url: "https://api.fireworks.ai/inference/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["transcription".into()],
                default_model: Some("whisper-v3".to_string()),
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "fireworks".to_string(),
            "https://api.fireworks.ai/inference/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        assert_eq!(spec.audio_base_url(&ctx), "https://audio.fireworks.ai/v1");
        assert!(spec.capabilities().supports("transcription"));
        assert!(spec.capabilities().supports("audio"));
        assert!(!spec.capabilities().supports("speech"));
    }

    #[test]
    fn openai_compatible_audio_base_url_prefers_explicit_ctx_override() {
        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "fireworks".to_string(),
                name: "Fireworks AI".to_string(),
                base_url: "https://api.fireworks.ai/inference/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["transcription".into()],
                default_model: Some("whisper-v3".to_string()),
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "fireworks".to_string(),
            "http://127.0.0.1:12345/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        assert_eq!(spec.audio_base_url(&ctx), "http://127.0.0.1:12345/v1");
    }

    #[test]
    fn openai_compatible_custom_provider_options_are_keyed_by_runtime_provider_id() {
        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::default().with_provider_option(
            "deepseek",
            serde_json::json!({
                "some_vendor_param": true
            }),
        );

        let hook = spec
            .chat_before_send(&req, &ctx)
            .expect("should install before_send for matching custom provider options");

        let body = serde_json::json!({
            "model": "deepseek-chat",
        });
        let out = hook(&body).unwrap();
        assert_eq!(
            out.get("some_vendor_param"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn openai_compatible_image_provider_options_merge_canonical_and_provider_owned_keys() {
        use crate::core::ProviderSpec;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["image_generation".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = ImageGenerationRequest::default()
            .with_provider_option(
                "openaiCompatible",
                serde_json::json!({ "user": "compat-user" }),
            )
            .with_provider_option(
                "deepseek",
                serde_json::json!({ "quality": "hd", "user": "provider-user" }),
            );

        let bundle = spec.choose_image_transformers(&req, &ctx);
        let body = bundle
            .request
            .transform_image(&req)
            .expect("transform image");

        assert_eq!(body["user"], serde_json::json!("provider-user"));
        assert_eq!(body["quality"], serde_json::json!("hd"));
    }

    #[test]
    fn openai_compatible_image_warnings_surface_ai_sdk_aspect_ratio_then_seed_on_generation() {
        use crate::core::ProviderSpec;

        let (spec, ctx) = compat_image_spec_and_ctx();
        let req = ImageGenerationRequest {
            aspect_ratio: Some("16:9".to_string()),
            seed: Some(7),
            ..Default::default()
        };

        assert_eq!(
            spec.image_warnings(&req, &ctx),
            Some(vec![
                Warning::unsupported(
                    "aspectRatio",
                    Some("This model does not support aspect ratio. Use `size` instead."),
                ),
                Warning::unsupported("seed", Option::<String>::None),
            ])
        );
    }

    #[test]
    fn openai_compatible_image_warnings_surface_ai_sdk_aspect_ratio_then_seed_on_edit() {
        use crate::core::ProviderSpec;

        let (spec, ctx) = compat_image_spec_and_ctx();
        let req = ImageEditRequest {
            images: vec![crate::types::ImageEditInput::file(vec![1, 2, 3])],
            mask: None,
            prompt: "edit".to_string(),
            model: None,
            count: None,
            size: None,
            aspect_ratio: Some("4:3".to_string()),
            seed: Some(9),
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        assert_eq!(
            spec.image_edit_warnings(&req, &ctx),
            Some(vec![
                Warning::unsupported(
                    "aspectRatio",
                    Some("This model does not support aspect ratio. Use `size` instead."),
                ),
                Warning::unsupported("seed", Option::<String>::None),
            ])
        );
    }

    #[test]
    fn openai_compatible_image_warnings_surface_ai_sdk_aspect_ratio_then_seed_on_variation() {
        use crate::core::ProviderSpec;

        let (spec, ctx) = compat_image_spec_and_ctx();
        let req = ImageVariationRequest {
            image: crate::types::ImageEditInput::file(vec![1, 2, 3]),
            model: None,
            count: None,
            size: None,
            aspect_ratio: Some("9:16".to_string()),
            seed: Some(11),
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };

        assert_eq!(
            spec.image_variation_warnings(&req, &ctx),
            Some(vec![
                Warning::unsupported(
                    "aspectRatio",
                    Some("This model does not support aspect ratio. Use `size` instead."),
                ),
                Warning::unsupported("seed", Option::<String>::None),
            ])
        );
    }

    #[test]
    fn openai_compatible_streaming_chat_omits_stream_options_by_default() {
        use crate::core::ProviderSpec;

        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: Default::default(),
            capabilities: vec!["tools".into()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let spec = OpenAiCompatibleSpecWithAdapter::new(adapter);

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn openai_compatible_include_usage_setting_restores_stream_options() {
        use crate::core::ProviderSpec;

        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: Default::default(),
            capabilities: vec!["tools".into()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            adapter,
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: Some(true),
                supports_structured_outputs: None,
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(
            body.get("stream_options"),
            Some(&serde_json::json!({ "include_usage": true }))
        );
    }

    #[test]
    fn openai_compatible_request_body_transformer_runs_after_include_usage_policy() {
        use crate::core::ProviderSpec;

        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: Default::default(),
            capabilities: vec!["tools".into()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            adapter,
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: Some(true),
                supports_structured_outputs: None,
                request_body_transformer: Some(Arc::new(
                    |body: &mut serde_json::Value, _model: &str, request_type| {
                        assert!(matches!(
                            request_type,
                            crate::standards::openai::compat::types::RequestType::Chat
                        ));
                        body.as_object_mut()
                            .expect("object body")
                            .remove("stream_options");
                        body["custom"] = serde_json::json!(true);
                        Ok(())
                    },
                )),
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .stream(true)
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert!(body.get("stream_options").is_none());
        assert_eq!(body.get("custom"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn openai_compatible_query_params_apply_to_all_compat_urls() {
        use crate::core::ProviderSpec;

        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: Default::default(),
            capabilities: vec!["chat".into(), "tools".into(), "image_generation".into()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            adapter,
            OpenAiCompatibleRequestSettings {
                query_params: std::collections::BTreeMap::from([
                    ("api-version".to_string(), "2025-04-01".to_string()),
                    ("tenant".to_string(), "acme".to_string()),
                ]),
                include_usage: None,
                supports_structured_outputs: None,
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let chat_req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .build();
        let embedding_req =
            crate::types::EmbeddingRequest::single("hi").with_model("text-embedding-3-small");
        let image_req = crate::types::ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            model: Some("deepseek-image".to_string()),
            ..Default::default()
        };
        let image_edit_req = ImageEditRequest {
            images: Vec::new(),
            mask: None,
            prompt: "edit".to_string(),
            model: None,
            count: None,
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };
        let image_variation_req = ImageVariationRequest {
            image: crate::types::ImageEditInput::file(Vec::new()),
            model: None,
            count: None,
            size: None,
            aspect_ratio: None,
            seed: None,
            response_format: None,
            extra_params: Default::default(),
            provider_options_map: Default::default(),
            http_config: None,
        };
        let rerank_req = RerankRequest::new(
            "rerank-model".to_string(),
            "query".to_string(),
            vec!["doc-1".to_string()],
        );

        assert_eq!(
            spec.chat_url(false, &chat_req, &ctx),
            "https://api.deepseek.com/v1/chat/completions?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.embedding_url(&embedding_req, &ctx),
            "https://api.deepseek.com/v1/embeddings?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.image_url(&image_req, &ctx),
            "https://api.deepseek.com/v1/images/generations?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.image_edit_url(&image_edit_req, &ctx),
            "https://api.deepseek.com/v1/images/edits?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.image_variation_url(&image_variation_req, &ctx),
            "https://api.deepseek.com/v1/images/variations?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.rerank_url(&rerank_req, &ctx),
            "https://api.deepseek.com/v1/rerank?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.models_url(&ctx),
            "https://api.deepseek.com/v1/models?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.model_url("deepseek-chat", &ctx),
            "https://api.deepseek.com/v1/models/deepseek-chat?api-version=2025-04-01&tenant=acme"
        );
        assert_eq!(
            spec.audio_base_url(&ctx),
            "https://api.deepseek.com/v1?api-version=2025-04-01&tenant=acme"
        );
    }

    #[test]
    fn openai_compatible_structured_outputs_policy_defaults_to_json_object() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: Default::default(),
            capabilities: vec!["tools".into()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            adapter,
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: None,
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(serde_json::json!({
                "type": "object",
                "properties": {}
            })))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({ "type": "json_object" }))
        );
    }

    #[test]
    fn openai_compatible_structured_outputs_policy_preserves_json_schema_when_enabled() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let adapter = Arc::new(ConfigurableAdapter::new(ProviderConfig {
            id: "deepseek".to_string(),
            name: "DeepSeek".to_string(),
            base_url: "https://api.deepseek.com/v1".to_string(),
            field_mappings: Default::default(),
            capabilities: vec!["tools".into()],
            default_model: None,
            supports_reasoning: true,
            api_key_env: None,
            api_key_env_aliases: vec![],
        }));
        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            adapter,
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: Some(true),
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_deepseek_runtime_provider_preserves_response_format_json_schema_when_enabled()
     {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            })),
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: Some(true),
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_deepseek_runtime_provider_preserves_tool_choice_none_and_tool_call_response_mapping()
     {
        use crate::core::ProviderSpec;
        use crate::types::{FinishReason, Tool, ToolChoice};

        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            })),
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: Some(true),
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "tool_choice": "auto"
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");
        assert_eq!(body.get("tool_choice"), Some(&serde_json::json!("none")));

        let raw = serde_json::json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });

        let resp = bundle
            .response
            .transform_chat_response(&raw)
            .expect("transform response");
        assert_eq!(resp.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(resp.tool_calls().len(), 1);
        let call = resp.tool_calls()[0].as_tool_call().expect("tool call");
        assert_eq!(call.tool_call_id, "call_1");
        assert_eq!(call.tool_name, "get_weather");
        assert_eq!(call.arguments, &serde_json::json!({ "city": "Tokyo" }));
    }

    #[test]
    fn openai_compatible_deepseek_runtime_provider_normalizes_reasoning_options_and_preserves_stable_fields()
     {
        use crate::core::ProviderSpec;
        use crate::types::{Tool, ToolChoice, chat::ResponseFormat};

        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            })),
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: Some(true),
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_provider_option(
                "deepseek",
                serde_json::json!({
                    "enableReasoning": true,
                    "reasoningBudget": 2048,
                    "response_format": { "type": "json_object" },
                    "tool_choice": "auto"
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["enable_reasoning"], serde_json::json!(true));
        assert_eq!(body["reasoning_budget"], serde_json::json!(2048));
        assert!(body.get("enableReasoning").is_none());
        assert!(body.get("reasoningBudget").is_none());
        assert_eq!(body.get("tool_choice"), Some(&serde_json::json!("none")));
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_openai_compatible_provider_options_apply_known_chat_fields() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "deepseek".to_string(),
                name: "DeepSeek".to_string(),
                base_url: "https://api.deepseek.com/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            })),
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: Some(true),
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "deepseek".to_string(),
            "https://api.deepseek.com/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let req = crate::types::ChatRequest::builder()
            .model("deepseek-chat")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_provider_option(
                "openaiCompatible",
                serde_json::json!({
                    "user": "compat-user",
                    "reasoningEffort": "high",
                    "textVerbosity": "medium",
                    "strictJsonSchema": false
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["user"], serde_json::json!("compat-user"));
        assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(body["verbosity"], serde_json::json!("medium"));
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": false
                }
            }))
        );
        assert!(body.get("strictJsonSchema").is_none());
        assert!(body.get("textVerbosity").is_none());
    }

    #[test]
    fn openai_compatible_xai_runtime_provider_preserves_response_format_json_schema_when_enabled() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "xai".to_string(),
                name: "xAI".to_string(),
                base_url: "https://api.x.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            })),
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: Some(true),
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "xai".to_string(),
            "https://api.x.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"],
            "additionalProperties": false
        });

        let req = crate::types::ChatRequest::builder()
            .model("grok-3-mini")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_xai_runtime_provider_keeps_stable_response_format_against_raw_provider_options()
     {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::with_settings(
            Arc::new(ConfigurableAdapter::new(ProviderConfig {
                id: "xai".to_string(),
                name: "xAI".to_string(),
                base_url: "https://api.x.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            })),
            OpenAiCompatibleRequestSettings {
                query_params: Default::default(),
                include_usage: None,
                supports_structured_outputs: Some(true),
                request_body_transformer: None,
            },
        );

        let ctx = ProviderContext::new(
            "xai".to_string(),
            "https://api.x.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });

        let req = crate::types::ChatRequest::builder()
            .model("grok-3-mini")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .stop_sequences(vec!["END".to_string()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "response_format": { "type": "json_object" },
                    "reasoningEffort": "high"
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert!(body.get("stop").is_none());
        assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": true
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_xai_runtime_provider_preserves_tool_choice_none_and_tool_call_response_mapping()
     {
        use crate::core::ProviderSpec;
        use crate::types::{FinishReason, Tool, ToolChoice};

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "xai".to_string(),
                name: "xAI".to_string(),
                base_url: "https://api.x.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "xai".to_string(),
            "https://api.x.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("grok-3-mini")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "tool_choice": "auto"
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");
        assert_eq!(body.get("tool_choice"), Some(&serde_json::json!("none")));

        let raw = serde_json::json!({
            "id": "chatcmpl_1",
            "object": "chat.completion",
            "created": 1,
            "model": "grok-3-mini",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": { "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3 }
        });

        let resp = bundle
            .response
            .transform_chat_response(&raw)
            .expect("transform response");
        assert_eq!(resp.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(resp.tool_calls().len(), 1);
        let call = resp.tool_calls()[0].as_tool_call().expect("tool call");
        assert_eq!(call.tool_call_id, "call_1");
        assert_eq!(call.tool_name, "get_weather");
        assert_eq!(call.arguments, &serde_json::json!({ "city": "Tokyo" }));
    }

    #[test]
    fn openai_compatible_xai_runtime_provider_normalizes_logprobs_and_drops_responses_only_fields()
    {
        use crate::core::ProviderSpec;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "xai".to_string(),
                name: "xAI".to_string(),
                base_url: "https://api.x.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "xai".to_string(),
            "https://api.x.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("grok-4")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .build()
            .with_provider_option(
                "xai",
                serde_json::json!({
                    "reasoningSummary": "detailed",
                    "topLogprobs": 2,
                    "previousResponseId": "resp_prev_123",
                    "include": ["file_search_call.results"],
                    "store": false
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["top_logprobs"], serde_json::json!(2));
        assert_eq!(body["logprobs"], serde_json::json!(true));
        assert!(body.get("reasoning_summary").is_none());
        assert!(body.get("previous_response_id").is_none());
        assert!(body.get("include").is_none());
        assert!(body.get("store").is_none());
        assert!(body.get("topLogprobs").is_none());
    }
}
