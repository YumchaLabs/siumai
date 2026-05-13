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
    let legacy_enable =
        take_any(obj, &["enableReasoning", "enable_reasoning"]).and_then(|value| value.as_bool());

    obj.remove("reasoningBudget");
    obj.remove("reasoning_budget");

    if let Some(thinking) = obj
        .get_mut("thinking")
        .and_then(|value| value.as_object_mut())
    {
        rename_field(thinking, "thinkingType", "type");
        rename_field(thinking, "thinking_type", "type");
        thinking.remove("budgetTokens");
        thinking.remove("budget_tokens");
    } else if let Some(enable) = legacy_enable {
        obj.insert(
            "thinking".to_string(),
            serde_json::json!({
                "type": if enable { "enabled" } else { "disabled" }
            }),
        );
    }
}

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

fn normalize_fireworks_reasoning_effort(value: &str) -> String {
    match value {
        "minimal" => "low".to_string(),
        "xhigh" => "high".to_string(),
        _ => value.to_string(),
    }
}

fn normalize_fireworks_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(value) = take_any(obj, &["reasoningHistory", "reasoning_history"]) {
        obj.entry("reasoning_history".to_string()).or_insert(value);
    }

    if let Some(thinking) = obj
        .get_mut("thinking")
        .and_then(|value| value.as_object_mut())
    {
        rename_field(thinking, "budgetTokens", "budget_tokens");
    }
}

fn normalize_moonshotai_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(value) = take_any(obj, &["reasoningHistory", "reasoning_history"]) {
        obj.entry("reasoning_history".to_string()).or_insert(value);
    }

    if let Some(thinking) = obj
        .get_mut("thinking")
        .and_then(|value| value.as_object_mut())
    {
        rename_field(thinking, "budgetTokens", "budget_tokens");
    }
}

fn normalize_alibaba_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(value) = take_any(obj, &["enableThinking", "enable_thinking"]) {
        obj.insert("enable_thinking".to_string(), value);
    }
    if let Some(value) = take_any(obj, &["thinkingBudget", "thinking_budget"]) {
        obj.insert("thinking_budget".to_string(), value);
    }
    if let Some(value) = take_any(obj, &["parallelToolCalls", "parallel_tool_calls"]) {
        obj.insert("parallel_tool_calls".to_string(), value);
    }
}

fn normalize_mistral_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(value) = take_any(obj, &["safePrompt", "safe_prompt"]) {
        obj.insert("safe_prompt".to_string(), value);
    }
    if let Some(value) = take_any(obj, &["documentImageLimit", "document_image_limit"]) {
        obj.insert("document_image_limit".to_string(), value);
    }
    if let Some(value) = take_any(obj, &["documentPageLimit", "document_page_limit"]) {
        obj.insert("document_page_limit".to_string(), value);
    }
}

fn normalize_perplexity_web_search_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
    if let Some(value) = take_any(obj, &["searchContextSize", "search_context_size"]) {
        obj.insert("search_context_size".to_string(), value);
    }

    if let Some(value) = take_any(obj, &["userLocation", "user_location"]) {
        obj.insert("user_location".to_string(), value);
    }
}

fn normalize_perplexity_options(obj: &mut serde_json::Map<String, serde_json::Value>) {
    for (aliases, canonical) in [
        (&["searchMode", "search_mode"][..], "search_mode"),
        (
            &["searchRecencyFilter", "search_recency_filter"][..],
            "search_recency_filter",
        ),
        (
            &["returnRelatedQuestions", "return_related_questions"][..],
            "return_related_questions",
        ),
        (&["returnImages", "return_images"][..], "return_images"),
        (&["disableSearch", "disable_search"][..], "disable_search"),
        (
            &["enableSearchClassifier", "enable_search_classifier"][..],
            "enable_search_classifier",
        ),
        (
            &["searchDomainFilter", "search_domain_filter"][..],
            "search_domain_filter",
        ),
        (
            &["searchLanguageFilter", "search_language_filter"][..],
            "search_language_filter",
        ),
        (
            &["searchAfterDateFilter", "search_after_date_filter"][..],
            "search_after_date_filter",
        ),
        (
            &["searchBeforeDateFilter", "search_before_date_filter"][..],
            "search_before_date_filter",
        ),
        (
            &["lastUpdatedAfterFilter", "last_updated_after_filter"][..],
            "last_updated_after_filter",
        ),
        (
            &["lastUpdatedBeforeFilter", "last_updated_before_filter"][..],
            "last_updated_before_filter",
        ),
        (
            &["imageDomainFilter", "image_domain_filter"][..],
            "image_domain_filter",
        ),
        (
            &["imageFormatFilter", "image_format_filter"][..],
            "image_format_filter",
        ),
    ] {
        if let Some(value) = take_any(obj, aliases) {
            obj.insert(canonical.to_string(), value);
        }
    }

    if let Some(mut value) = take_any(obj, &["webSearchOptions", "web_search_options"]) {
        if let Some(web_search) = value.as_object_mut() {
            normalize_perplexity_web_search_options(web_search);
        }
        obj.insert("web_search_options".to_string(), value);
    }
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
    structured_outputs: Option<bool>,
    parallel_tool_calls: Option<bool>,
}

fn compat_chat_options(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> CompatChatOptions {
    let mut out = CompatChatOptions::default();

    for options in [Some("openai-compatible"), Some("openaiCompatible")]
        .into_iter()
        .flatten()
        .filter_map(|key| map.get_object(key))
        .chain(
            crate::standards::openai::compat::metadata::provider_options_keys(provider_id)
                .into_iter()
                .filter_map(|key| map.get_object(&key)),
        )
    {
        if let Some(user) = string_option_by_aliases(options, &["user"]) {
            out.user = Some(user);
        }
        if let Some(reasoning_effort) =
            string_option_by_aliases(options, &["reasoningEffort", "reasoning_effort"])
        {
            out.reasoning_effort = Some(if provider_id == "fireworks" {
                normalize_fireworks_reasoning_effort(&reasoning_effort)
            } else {
                reasoning_effort
            });
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
        if let Some(structured_outputs) =
            bool_option_by_aliases(options, &["structuredOutputs", "structured_outputs"])
        {
            out.structured_outputs = Some(structured_outputs);
        }
        if let Some(parallel_tool_calls) =
            bool_option_by_aliases(options, &["parallelToolCalls", "parallel_tool_calls"])
        {
            out.parallel_tool_calls = Some(parallel_tool_calls);
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
        "structuredOutputs",
        "structured_outputs",
        "parallelToolCalls",
        "parallel_tool_calls",
    ] {
        obj.remove(key);
    }
}

fn compat_image_provider_options(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut merged = serde_json::Map::new();

    for options in [Some("openai-compatible"), Some("openaiCompatible")]
        .into_iter()
        .flatten()
        .filter_map(|key| map.get_object(key))
        .chain(
            crate::standards::openai::compat::metadata::provider_options_keys(provider_id)
                .into_iter()
                .filter_map(|key| map.get_object(&key)),
        )
    {
        for (key, value) in options {
            merged.insert(key.clone(), value.clone());
        }
    }

    (!merged.is_empty()).then_some(merged)
}

fn is_togetherai_provider(provider_id: &str) -> bool {
    matches!(provider_id, "together" | "togetherai")
}

fn parse_image_size(size: &str) -> Option<(u32, u32)> {
    let (width, height) = size.split_once('x').or_else(|| size.split_once('X'))?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
}

fn normalize_togetherai_image_generation_request(
    req: &ImageGenerationRequest,
    body: &mut serde_json::Value,
    provider_options: Option<&serde_json::Map<String, serde_json::Value>>,
) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    let provider_has = |key: &str| {
        provider_options
            .as_ref()
            .is_some_and(|options| options.contains_key(key))
    };

    if let Some((width, height)) = req.size.as_deref().and_then(parse_image_size) {
        obj.insert("width".to_string(), serde_json::json!(width));
        obj.insert("height".to_string(), serde_json::json!(height));
    }
    if !provider_has("size") {
        obj.remove("size");
    }

    if req.count <= 1 && !provider_has("n") {
        obj.remove("n");
    }

    if let Some(seed) = req.seed {
        obj.insert("seed".to_string(), serde_json::json!(seed));
    }

    if !provider_has("response_format") {
        obj.insert("response_format".to_string(), serde_json::json!("base64"));
    }
    if !provider_has("quality") {
        obj.remove("quality");
    }
    if !provider_has("style") {
        obj.remove("style");
    }
}

fn normalize_provider_options(
    provider_id: &str,
    value: Option<&serde_json::Value>,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut obj = value?.as_object()?.clone();

    if provider_id == "xai" {
        if let Some(value) = take_any(&mut obj, &["reasoningEffort", "reasoning_effort"]) {
            obj.entry("reasoning_effort".to_string()).or_insert(value);
        }
        rename_field(&mut obj, "reasoningSummary", "reasoning_summary");
        rename_field(&mut obj, "searchParameters", "search_parameters");
        rename_field(&mut obj, "topLogprobs", "top_logprobs");
        rename_field(&mut obj, "previousResponseId", "previous_response_id");
        let legacy_reasoning_enabled = take_any(
            &mut obj,
            &[
                "enableReasoning",
                "enable_reasoning",
                "enableThinking",
                "enable_thinking",
            ],
        )
        .and_then(|value| value.as_bool());
        let legacy_reasoning_budget = take_any(
            &mut obj,
            &[
                "reasoningBudget",
                "reasoning_budget",
                "thinkingBudget",
                "thinking_budget",
            ],
        );
        if obj.get("reasoning_effort").is_none() {
            if legacy_reasoning_budget.is_some() {
                obj.insert("reasoning_effort".to_string(), serde_json::json!("high"));
            } else if matches!(legacy_reasoning_enabled, Some(true)) {
                obj.insert("reasoning_effort".to_string(), serde_json::json!("low"));
            }
        }
        if let Some(v) = obj.get_mut("search_parameters") {
            normalize_xai_search_parameters(v);
        }
        if obj.get("top_logprobs").is_some() {
            obj.insert("logprobs".to_string(), serde_json::json!(true));
        }
    } else if provider_id == "deepseek" {
        normalize_deepseek_options(&mut obj);
    } else if provider_id == "mistral" {
        normalize_mistral_options(&mut obj);
    } else if provider_id == "perplexity" {
        normalize_perplexity_options(&mut obj);
    } else if provider_id == "fireworks" {
        normalize_fireworks_options(&mut obj);
    } else if provider_id == "moonshotai" || provider_id == "moonshot" {
        normalize_moonshotai_options(&mut obj);
    } else if provider_id == "qwen" || provider_id == "alibaba" {
        normalize_alibaba_options(&mut obj);
    } else if provider_id == "groq" {
        if let Some(value) = take_any(&mut obj, &["max_completion_tokens", "max_tokens"]) {
            obj.entry("max_tokens".to_string()).or_insert(value);
        }
        rename_field(&mut obj, "serviceTier", "service_tier");
        rename_field(&mut obj, "reasoningFormat", "reasoning_format");
        rename_field(&mut obj, "topLogprobs", "top_logprobs");
    }

    Some(obj)
}

fn merged_normalized_provider_options(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut merged = serde_json::Map::new();

    for options in crate::standards::openai::compat::metadata::provider_options_keys(provider_id)
        .into_iter()
        .filter_map(|key| map.get(&key))
        .filter_map(|value| normalize_provider_options(provider_id, Some(value)))
    {
        for (key, value) in options {
            merged.insert(key, value);
        }
    }

    (!merged.is_empty()).then_some(merged)
}

fn remove_known_provider_chat_option_keys(
    provider_id: &str,
    obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    remove_known_compat_chat_option_keys(obj);

    if matches!(provider_id, "qwen" | "alibaba") {
        obj.remove("cacheControl");
        obj.remove("cache_control");
    }
}

fn normalize_chat_passthrough_provider_options(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut obj = merged_normalized_provider_options(provider_id, map)?;
    remove_known_provider_chat_option_keys(provider_id, &mut obj);
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

    if let Some(parallel_tool_calls) = compat_options.parallel_tool_calls
        && body_obj.get("tools").is_some()
    {
        body_obj.insert(
            "parallel_tool_calls".to_string(),
            serde_json::Value::Bool(parallel_tool_calls),
        );
    }
}

fn apply_chat_request_settings(
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
    settings: &OpenAiCompatibleRequestSettings,
    structured_outputs_override: Option<bool>,
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

    let supports_structured_outputs =
        structured_outputs_override.or(settings.supports_structured_outputs);

    if supports_structured_outputs != Some(true) && request_uses_structured_outputs {
        body_obj.insert(
            "response_format".to_string(),
            serde_json::json!({ "type": "json_object" }),
        );
    }
}

fn mistral_json_instruction(existing: Option<&str>) -> String {
    match existing.filter(|value| !value.is_empty()) {
        Some(prompt) => format!("{prompt}\n\nYou MUST answer with JSON."),
        None => "You MUST answer with JSON.".to_string(),
    }
}

fn apply_mistral_json_object_instruction(
    provider_id: &str,
    req: &crate::types::ChatRequest,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    if provider_id != "mistral"
        || !matches!(
            req.response_format,
            Some(crate::types::chat::ResponseFormat::JsonObject { .. })
        )
    {
        return;
    }

    let Some(messages) = body_obj
        .get_mut("messages")
        .and_then(|value| value.as_array_mut())
    else {
        return;
    };

    if let Some(first) = messages.first_mut()
        && first.get("role").and_then(|value| value.as_str()) == Some("system")
        && let Some(content) = first.get_mut("content")
        && let Some(text) = content.as_str()
    {
        *content = serde_json::Value::String(mistral_json_instruction(Some(text)));
        return;
    }

    messages.insert(
        0,
        serde_json::json!({
            "role": "system",
            "content": mistral_json_instruction(None),
        }),
    );
}

fn passthrough_provider_option_has(
    provider_options: &Option<serde_json::Map<String, serde_json::Value>>,
    key: &str,
) -> bool {
    provider_options
        .as_ref()
        .is_some_and(|options| options.contains_key(key))
}

fn set_json_schema_strict(body_obj: &mut serde_json::Map<String, serde_json::Value>, strict: bool) {
    if let Some(json_schema_obj) = body_obj
        .get_mut("response_format")
        .and_then(|value| value.as_object_mut())
        .filter(|obj| obj.get("type").and_then(|value| value.as_str()) == Some("json_schema"))
        .and_then(|obj| obj.get_mut("json_schema"))
        .and_then(|value| value.as_object_mut())
    {
        json_schema_obj.insert("strict".to_string(), serde_json::Value::Bool(strict));
    }
}

fn openai_chat_function_tool_name(tool: &serde_json::Value) -> Option<&str> {
    tool.get("function")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
}

fn apply_mistral_tool_choice(
    req: &crate::types::ChatRequest,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    let Some(choice) = req.tool_choice.as_ref() else {
        return;
    };

    if !body_obj.get("tools").is_some_and(|value| value.is_array()) {
        return;
    }

    match choice {
        crate::types::ToolChoice::Required => {
            body_obj.insert("tool_choice".to_string(), serde_json::json!("any"));
        }
        crate::types::ToolChoice::Tool { name } => {
            if let Some(tools) = body_obj
                .get_mut("tools")
                .and_then(|value| value.as_array_mut())
            {
                tools.retain(|tool| openai_chat_function_tool_name(tool) == Some(name.as_str()));
            }
            body_obj.insert("tool_choice".to_string(), serde_json::json!("any"));
        }
        crate::types::ToolChoice::Auto | crate::types::ToolChoice::None => {}
    }
}

fn apply_xai_chat_settings(
    provider_id: &str,
    req: &crate::types::ChatRequest,
    provider_options: &Option<serde_json::Map<String, serde_json::Value>>,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    if provider_id != "xai" {
        return;
    }

    if req.common_params.frequency_penalty.is_some()
        && !passthrough_provider_option_has(provider_options, "frequency_penalty")
    {
        body_obj.remove("frequency_penalty");
    }

    if req.common_params.presence_penalty.is_some()
        && !passthrough_provider_option_has(provider_options, "presence_penalty")
    {
        body_obj.remove("presence_penalty");
    }

    if req.common_params.stop_sequences.is_some()
        && !passthrough_provider_option_has(provider_options, "stop")
    {
        body_obj.remove("stop");
    }

    body_obj.remove("reasoning_summary");
    body_obj.remove("previous_response_id");
    body_obj.remove("include");
    body_obj.remove("store");
    set_json_schema_strict(body_obj, true);
}

fn apply_deepseek_chat_settings(
    provider_id: &str,
    req: &crate::types::ChatRequest,
    provider_options: &Option<serde_json::Map<String, serde_json::Value>>,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    if provider_id != "deepseek" {
        return;
    }

    if req.common_params.seed.is_some()
        && !passthrough_provider_option_has(provider_options, "seed")
    {
        body_obj.remove("seed");
    }

    if req.response_format.is_some() {
        body_obj.insert(
            "response_format".to_string(),
            serde_json::json!({ "type": "json_object" }),
        );
    }
}

fn apply_mistral_chat_settings(
    provider_id: &str,
    req: &crate::types::ChatRequest,
    provider_options: &Option<serde_json::Map<String, serde_json::Value>>,
    compat_options: &CompatChatOptions,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    if provider_id != "mistral" {
        return;
    }

    apply_mistral_tool_choice(req, body_obj);

    if req.common_params.frequency_penalty.is_some()
        && !passthrough_provider_option_has(provider_options, "frequency_penalty")
    {
        body_obj.remove("frequency_penalty");
    }

    if req.common_params.presence_penalty.is_some()
        && !passthrough_provider_option_has(provider_options, "presence_penalty")
    {
        body_obj.remove("presence_penalty");
    }

    if req.common_params.stop_sequences.is_some()
        && !passthrough_provider_option_has(provider_options, "stop")
    {
        body_obj.remove("stop");
    }

    if let Some(seed) = req.common_params.seed {
        if !passthrough_provider_option_has(provider_options, "random_seed") {
            body_obj.insert("random_seed".to_string(), serde_json::json!(seed));
        }
        if !passthrough_provider_option_has(provider_options, "seed") {
            body_obj.remove("seed");
        }
    }

    if compat_options.strict_json_schema.is_none()
        && let Some(crate::types::chat::ResponseFormat::Json { strict, .. }) =
            req.response_format.as_ref()
        && strict.is_none()
    {
        set_json_schema_strict(body_obj, false);
    }

    let model = body_obj
        .get("model")
        .and_then(|value| value.as_str())
        .unwrap_or(req.common_params.model.as_str());
    if !matches!(model, "mistral-small-latest" | "mistral-small-2603") {
        body_obj.remove("reasoning_effort");
    }
}

fn apply_perplexity_chat_settings(
    provider_id: &str,
    req: &crate::types::ChatRequest,
    provider_options: &Option<serde_json::Map<String, serde_json::Value>>,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    if provider_id != "perplexity" {
        return;
    }

    if let Some(top_k) = req.common_params.top_k
        && !passthrough_provider_option_has(provider_options, "top_k")
    {
        body_obj.insert("top_k".to_string(), serde_json::json!(top_k));
    }

    if req.common_params.seed.is_some()
        && !passthrough_provider_option_has(provider_options, "seed")
    {
        body_obj.remove("seed");
    }

    if req.common_params.stop_sequences.is_some()
        && !passthrough_provider_option_has(provider_options, "stop")
    {
        body_obj.remove("stop");
    }

    if let Some(response_format) = req.response_format.as_ref() {
        let value = match response_format {
            crate::types::chat::ResponseFormat::JsonObject { .. } => serde_json::json!({
                "type": "json_schema",
                "json_schema": {},
            }),
            crate::types::chat::ResponseFormat::Json { .. } => {
                crate::standards::openai::utils::convert_chat_completions_response_format(
                    response_format,
                    true,
                )
            }
        };

        body_obj.insert("response_format".to_string(), value);
    }
}

fn apply_alibaba_chat_settings(
    provider_id: &str,
    req: &crate::types::ChatRequest,
    provider_options: &Option<serde_json::Map<String, serde_json::Value>>,
    body_obj: &mut serde_json::Map<String, serde_json::Value>,
) {
    if !matches!(provider_id, "qwen" | "alibaba") {
        return;
    }

    if let Some(top_k) = req.common_params.top_k
        && !passthrough_provider_option_has(provider_options, "top_k")
    {
        body_obj.insert("top_k".to_string(), serde_json::json!(top_k));
    }

    if req.common_params.frequency_penalty.is_some()
        && !passthrough_provider_option_has(provider_options, "frequency_penalty")
    {
        body_obj.remove("frequency_penalty");
    }

    let Some(response_format) = req.response_format.as_ref() else {
        return;
    };

    let value = match response_format {
        crate::types::chat::ResponseFormat::JsonObject { .. } => {
            serde_json::json!({ "type": "json_object" })
        }
        crate::types::chat::ResponseFormat::Json {
            schema,
            name,
            description,
            ..
        } => {
            let mut json_schema = serde_json::Map::new();
            json_schema.insert("schema".to_string(), schema.clone());
            json_schema.insert(
                "name".to_string(),
                serde_json::json!(name.as_deref().unwrap_or("response")),
            );
            if let Some(description) = description.as_deref() {
                json_schema.insert("description".to_string(), serde_json::json!(description));
            }

            serde_json::json!({
                "type": "json_schema",
                "json_schema": json_schema,
            })
        }
    };

    body_obj.insert("response_format".to_string(), value);
}

fn provider_options_map_merge_hook(
    provider_id: &str,
    map: &crate::types::ProviderOptionsMap,
) -> Option<crate::execution::executors::BeforeSendHook> {
    let provider_id = provider_id.to_string();
    let provider_options = merged_normalized_provider_options(&provider_id, map);
    provider_options.as_ref()?;
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
    req: &crate::types::ChatRequest,
    settings: &OpenAiCompatibleRequestSettings,
    supports_stream_usage_hints: bool,
    request_uses_structured_outputs: bool,
) -> crate::execution::executors::BeforeSendHook {
    let provider_id = provider_id.to_string();
    let map = &req.provider_options_map;
    let compat_options = compat_chat_options(&provider_id, map);
    let provider_options = normalize_chat_passthrough_provider_options(&provider_id, map);
    let req = req.clone();
    let settings = settings.clone();
    Arc::new(
        move |body: &serde_json::Value| -> Result<serde_json::Value, LlmError> {
            let mut out = body.clone();
            if let Some(body_obj) = out.as_object_mut() {
                apply_compat_chat_options(body_obj, &compat_options);
                merge_provider_options_into_body(&provider_id, &provider_options, body_obj);
                super::alibaba_cache_control::apply_cache_controls_to_chat_body(
                    &provider_id,
                    &req,
                    body_obj,
                );
                apply_mistral_json_object_instruction(&provider_id, &req, body_obj);
                apply_chat_request_settings(
                    body_obj,
                    &settings,
                    compat_options.structured_outputs,
                    supports_stream_usage_hints,
                    request_uses_structured_outputs,
                );
                apply_perplexity_chat_settings(&provider_id, &req, &provider_options, body_obj);
                apply_alibaba_chat_settings(&provider_id, &req, &provider_options, body_obj);
                apply_xai_chat_settings(&provider_id, &req, &provider_options, body_obj);
                apply_deepseek_chat_settings(&provider_id, &req, &provider_options, body_obj);
                apply_mistral_chat_settings(
                    &provider_id,
                    &req,
                    &provider_options,
                    &compat_options,
                    body_obj,
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

fn default_request_settings_for_provider(provider_id: &str) -> OpenAiCompatibleRequestSettings {
    OpenAiCompatibleRequestSettings {
        supports_structured_outputs: match provider_id {
            // Keep this aligned with the compat config defaults used by the public client surface.
            "openrouter" | "perplexity" | "mistral" | "groq" | "qwen" | "alibaba" | "xai" => {
                Some(true)
            }
            _ => None,
        },
        ..OpenAiCompatibleRequestSettings::default()
    }
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
        let provider_id = adapter.provider_id().into_owned();
        Self::with_settings(adapter, default_request_settings_for_provider(&provider_id))
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

    /// Build the OpenAI-compatible completion endpoint URL.
    ///
    /// This mirrors the audited AI SDK `completionModel()` provider route and is primarily kept as
    /// a lower-contract/public-fixture audit helper for completion-capable compat providers.
    pub fn completion_url(&self, ctx: &ProviderContext) -> String {
        apply_url_settings(
            format!("{}{}", ctx.base_url.trim_end_matches('/'), "/completions"),
            &self.request_settings,
        )
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
            req,
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
        _req: &crate::types::ImageGenerationRequest,
        _ctx: &ProviderContext,
    ) -> ImageTransformers {
        let transformers = self
            .image_standard()
            .create_transformers(self.adapter.provider_id().as_ref());
        ImageTransformers {
            request: transformers.request,
            response: transformers.response,
        }
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
        let provider_id = self.adapter.provider_id();

        if req.aspect_ratio.is_some() {
            warnings.push(Warning::unsupported(
                "aspectRatio",
                Some("This model does not support aspect ratio. Use `size` instead."),
            ));
        }

        if req.seed.is_some() && !is_togetherai_provider(provider_id.as_ref()) {
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
        let provider_id = self.adapter.provider_id();

        if req.aspect_ratio.is_some() {
            warnings.push(Warning::unsupported(
                "aspectRatio",
                Some("This model does not support aspect ratio. Use `size` instead."),
            ));
        }

        if req.seed.is_some() && !is_togetherai_provider(provider_id.as_ref()) {
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
        let provider_id = self.adapter.provider_id();

        if req.aspect_ratio.is_some() {
            warnings.push(Warning::unsupported(
                "aspectRatio",
                Some("This model does not support aspect ratio. Use `size` instead."),
            ));
        }

        if req.seed.is_some() && !is_togetherai_provider(provider_id.as_ref()) {
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
                if api_key.is_empty() && !base_headers.contains_key(reqwest::header::AUTHORIZATION)
                {
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
        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::ImageGeneration)
                .trim_start_matches('/')
        );

        self.image_standard_with_endpoint(endpoint)
            .create_spec("openai_compatible")
    }

    fn image_standard(&self) -> crate::standards::openai::image::OpenAiImageStandard {
        let endpoint = format!(
            "/{}",
            self.adapter
                .route_for(super::types::RequestType::ImageGeneration)
                .trim_start_matches('/')
        );

        self.image_standard_with_endpoint(endpoint)
    }

    fn image_standard_with_endpoint(
        &self,
        endpoint: String,
    ) -> crate::standards::openai::image::OpenAiImageStandard {
        #[derive(Clone)]
        struct CompatImageAdapter {
            adapter: Arc<dyn super::adapter::ProviderAdapter>,
            generation_endpoint: String,
        }

        impl crate::standards::openai::image::OpenAiImageAdapter for CompatImageAdapter {
            fn transform_generation_request(
                &self,
                req: &crate::types::ImageGenerationRequest,
                body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                let model_s = body
                    .get("model")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                let provider_id = self.adapter.provider_id();
                let provider_options =
                    compat_image_provider_options(provider_id.as_ref(), &req.provider_options_map);
                if is_togetherai_provider(provider_id.as_ref()) {
                    normalize_togetherai_image_generation_request(
                        req,
                        body,
                        provider_options.as_ref(),
                    );
                }
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

        crate::standards::openai::image::OpenAiImageStandard::with_adapter(Arc::new(
            CompatImageAdapter {
                adapter: self.adapter.clone(),
                generation_endpoint: endpoint,
            },
        ))
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
    fn openai_compatible_chat_headers_allow_preexisting_authorization() {
        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "vertex-maas".to_string(),
                name: "Vertex MaaS".to_string(),
                base_url:
                    "https://aiplatform.googleapis.com/v1/projects/demo/locations/global/endpoints/openapi"
                        .to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into(), "embedding".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let mut extra_headers = std::collections::HashMap::new();
        extra_headers.insert(
            "Authorization".to_string(),
            "Bearer google-token".to_string(),
        );
        let ctx = ProviderContext::new(
            "vertex-maas".to_string(),
            "https://aiplatform.googleapis.com/v1/projects/demo/locations/global/endpoints/openapi"
                .to_string(),
            None,
            extra_headers,
        );

        let headers = spec
            .chat_spec()
            .build_headers(&ctx)
            .expect("Authorization header should satisfy compat auth");

        assert_eq!(
            headers
                .get(reqwest::header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok()),
            Some("Bearer google-token")
        );
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
    fn openai_compatible_image_response_metadata_uses_runtime_provider_key() {
        use crate::core::ProviderSpec;

        let (spec, ctx) = compat_image_spec_and_ctx();
        let req = ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            model: Some("deepseek-image".to_string()),
            ..Default::default()
        };

        let bundle = spec.choose_image_transformers(&req, &ctx);
        let response = bundle
            .response
            .transform_image_response(&serde_json::json!({
                "data": [{ "b64_json": "base64encodeddata" }]
            }))
            .expect("transform image response");

        assert!(response.metadata.contains_key("deepseek"));
        assert!(!response.metadata.contains_key("openai"));
        assert!(!response.metadata.contains_key("openai_compatible"));
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
            id: "openrouter".to_string(),
            name: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
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
            "openrouter".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
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
            .model("openai/gpt-4o")
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
    fn openai_compatible_deepseek_runtime_provider_uses_json_object_response_format() {
        use crate::core::ProviderSpec;
        use crate::types::{CommonParams, chat::ResponseFormat};

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
            .model_params(CommonParams {
                model: "deepseek-chat".to_string(),
                seed: Some(1234),
                ..CommonParams::default()
            })
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert!(body.get("seed").is_none());
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({ "type": "json_object" }))
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
    fn openai_compatible_deepseek_runtime_normalizes_thinking_options() {
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

        assert_eq!(
            body["thinking"],
            serde_json::json!({
                "type": "enabled"
            })
        );
        assert!(body.get("enableReasoning").is_none());
        assert!(body.get("enable_reasoning").is_none());
        assert!(body.get("reasoningBudget").is_none());
        assert!(body.get("reasoning_budget").is_none());
        assert_eq!(body.get("tool_choice"), Some(&serde_json::json!("none")));
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({ "type": "json_object" }))
        );
    }

    #[test]
    fn openai_compatible_mistral_chat_settings_follow_ai_sdk_body_shape() {
        use crate::core::ProviderSpec;
        use crate::types::{CommonParams, chat::ResponseFormat};

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "mistral".to_string(),
                name: "Mistral AI".to_string(),
                base_url: "https://api.mistral.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "mistral".to_string(),
            "https://api.mistral.ai/v1".to_string(),
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
            .model_params(CommonParams {
                model: "mistral-large-latest".to_string(),
                frequency_penalty: Some(0.2),
                presence_penalty: Some(0.4),
                stop_sequences: Some(vec!["END".to_string()]),
                seed: Some(99),
                ..CommonParams::default()
            })
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("custom"))
            .build()
            .with_provider_option("mistral", serde_json::json!({ "reasoningEffort": "high" }));

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
        assert!(body.get("stop").is_none());
        assert!(body.get("seed").is_none());
        assert_eq!(body.get("random_seed"), Some(&serde_json::json!(99)));
        assert!(body.get("reasoning_effort").is_none());
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "custom",
                    "schema": schema,
                    "strict": false
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
                id: "openrouter".to_string(),
                name: "OpenRouter".to_string(),
                base_url: "https://openrouter.ai/api/v1".to_string(),
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
            "openrouter".to_string(),
            "https://openrouter.ai/api/v1".to_string(),
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
            .model("openai/gpt-4o")
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
    fn fireworks_reasoning_effort_levels_follow_ai_sdk_mapping() {
        assert_eq!(normalize_fireworks_reasoning_effort("minimal"), "low");
        assert_eq!(normalize_fireworks_reasoning_effort("low"), "low");
        assert_eq!(normalize_fireworks_reasoning_effort("high"), "high");
        assert_eq!(normalize_fireworks_reasoning_effort("xhigh"), "high");
    }

    #[test]
    fn openai_compatible_qwen_defaults_to_json_schema_response_format() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "qwen".to_string(),
                name: "Qwen".to_string(),
                base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["chat".into(), "streaming".into(), "tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "qwen".to_string(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
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
            .model("qwen-plus")
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
                    "schema": schema
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_qwen_chat_settings_follow_alibaba_provider_shape() {
        use crate::core::ProviderSpec;
        use crate::types::{CommonParams, chat::ResponseFormat};

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "qwen".to_string(),
                name: "Qwen".to_string(),
                base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["chat".into(), "streaming".into(), "tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "qwen".to_string(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
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
            .model_params(CommonParams {
                model: "qwen-plus".to_string(),
                top_k: Some(20.0),
                frequency_penalty: Some(0.2),
                presence_penalty: Some(0.4),
                ..CommonParams::default()
            })
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(
                ResponseFormat::json_schema(schema.clone())
                    .with_name("custom")
                    .with_description("kept")
                    .with_strict(false),
            )
            .build()
            .with_provider_option(
                "alibaba",
                serde_json::json!({ "strictJsonSchema": true, "structuredOutputs": false }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body.get("top_k"), Some(&serde_json::json!(20.0)));
        assert!(body.get("frequency_penalty").is_none());
        assert_eq!(body.get("presence_penalty"), Some(&serde_json::json!(0.4)));
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "custom",
                    "description": "kept",
                    "schema": schema
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_qwen_runtime_accepts_alibaba_provider_options() {
        use crate::core::ProviderSpec;
        use crate::types::Tool;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "qwen".to_string(),
                name: "Qwen".to_string(),
                base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["chat".into(), "streaming".into(), "tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "qwen".to_string(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("qwen-plus")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "lookup",
                "Lookup value",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .build()
            .with_provider_option(
                "alibaba",
                serde_json::json!({
                    "enableThinking": true,
                    "thinkingBudget": 2048,
                    "parallelToolCalls": false
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["enable_thinking"], serde_json::json!(true));
        assert_eq!(body["thinking_budget"], serde_json::json!(2048));
        assert_eq!(body["parallel_tool_calls"], serde_json::json!(false));
        assert!(body.get("enableThinking").is_none());
        assert!(body.get("thinkingBudget").is_none());
        assert!(body.get("parallelToolCalls").is_none());
    }

    #[test]
    fn openai_compatible_qwen_runtime_applies_alibaba_prompt_cache_control() {
        use crate::core::ProviderSpec;
        use crate::types::{
            ChatMessage, ContentPart, MessageContent, MessageMetadata, MessageRole,
            ProviderOptionsMap,
        };

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "qwen".to_string(),
                name: "Qwen".to_string(),
                base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["chat".into(), "streaming".into(), "tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "qwen".to_string(),
            "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let mut tool_message_options = ProviderOptionsMap::default();
        tool_message_options.insert(
            "alibaba",
            serde_json::json!({ "cacheControl": { "type": "tool-message" } }),
        );

        let req = crate::types::ChatRequest::builder()
            .model("qwen-plus")
            .messages(vec![
                ChatMessage::system("system prompt")
                    .with_provider_option(
                        "alibaba",
                        serde_json::json!({ "cacheControl": { "type": "system" } }),
                    )
                    .build(),
                ChatMessage::user("")
                    .with_content_parts(vec![
                        ContentPart::text("look").with_provider_option(
                            "qwen",
                            serde_json::json!({ "cache_control": { "type": "user-part" } }),
                        ),
                        ContentPart::image_url("https://example.com/a.png"),
                    ])
                    .with_provider_option(
                        "alibaba",
                        serde_json::json!({ "cacheControl": { "type": "user-message" } }),
                    )
                    .build(),
                ChatMessage::assistant("answer")
                    .with_provider_option(
                        "qwen",
                        serde_json::json!({ "cache_control": { "type": "assistant" } }),
                    )
                    .build(),
                ChatMessage {
                    role: MessageRole::Tool,
                    content: MessageContent::MultiModal(vec![
                        ContentPart::tool_result_text("call_1", "lookup", "first")
                            .with_provider_option(
                                "qwen",
                                serde_json::json!({ "cacheControl": { "type": "tool-part" } }),
                            ),
                        ContentPart::tool_result_text("call_2", "lookup", "second"),
                    ]),
                    provider_options: tool_message_options,
                    metadata: MessageMetadata::default(),
                },
            ])
            .build()
            .with_provider_option(
                "alibaba",
                serde_json::json!({
                    "enableThinking": true,
                    "cacheControl": { "type": "request-level-is-not-a-body-param" }
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["enable_thinking"], serde_json::json!(true));
        assert!(body.get("cacheControl").is_none());

        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"],
            serde_json::json!({ "type": "system" })
        );
        assert_eq!(
            body["messages"][1]["content"][0]["cache_control"],
            serde_json::json!({ "type": "user-part" })
        );
        assert_eq!(
            body["messages"][1]["content"][1]["cache_control"],
            serde_json::json!({ "type": "user-message" })
        );
        assert_eq!(
            body["messages"][2]["content"][0]["cache_control"],
            serde_json::json!({ "type": "assistant" })
        );
        assert_eq!(
            body["messages"][3]["content"][0]["cache_control"],
            serde_json::json!({ "type": "tool-part" })
        );
        assert_eq!(
            body["messages"][4]["content"][0]["cache_control"],
            serde_json::json!({ "type": "tool-message" })
        );

        let warnings =
            crate::standards::openai::compat::alibaba_cache_control::cache_control_warnings(
                "qwen", &req,
            );
        assert_eq!(warnings.len(), 2);
        assert!(warnings.iter().all(|warning| {
            matches!(
                warning,
                Warning::Other { message }
                    if message
                        == crate::standards::openai::compat::alibaba_cache_control::CACHE_BREAKPOINT_LIMIT_WARNING
            )
        }));
    }

    #[test]
    fn openai_compatible_fireworks_runtime_provider_normalizes_fireworks_chat_options() {
        use crate::core::ProviderSpec;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "fireworks".to_string(),
                name: "Fireworks".to_string(),
                base_url: "https://api.fireworks.ai/inference/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["chat".into(), "streaming".into(), "tools".into()],
                default_model: None,
                supports_reasoning: true,
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

        let req = crate::types::ChatRequest::builder()
            .model("accounts/fireworks/models/llama-v3p1-8b-instruct")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .build()
            .with_provider_option(
                "fireworks",
                serde_json::json!({
                    "reasoningEffort": "xhigh",
                    "reasoningHistory": "preserved",
                    "thinking": {
                        "type": "enabled",
                        "budgetTokens": 4096
                    }
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
        assert_eq!(body["reasoning_history"], serde_json::json!("preserved"));
        assert_eq!(
            body["thinking"],
            serde_json::json!({
                "type": "enabled",
                "budget_tokens": 4096
            })
        );
        assert!(body.get("reasoningHistory").is_none());
        assert!(body["thinking"].get("budgetTokens").is_none());
    }

    #[test]
    fn openai_compatible_mistral_defaults_to_structured_outputs_enabled() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "mistral".to_string(),
                name: "Mistral AI".to_string(),
                base_url: "https://api.mistral.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "mistral".to_string(),
            "https://api.mistral.ai/v1".to_string(),
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
            .model("mistral-large-latest")
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
                    "strict": false
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_mistral_json_object_injects_json_instruction() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "mistral".to_string(),
                name: "Mistral AI".to_string(),
                base_url: "https://api.mistral.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "mistral".to_string(),
            "https://api.mistral.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model("mistral-small-latest")
            .messages(vec![
                crate::types::ChatMessage::system("Existing system").build(),
                crate::types::ChatMessage::user("hi").build(),
            ])
            .response_format(ResponseFormat::json_object())
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(
            body["messages"][0],
            serde_json::json!({
                "role": "system",
                "content": "Existing system\n\nYou MUST answer with JSON."
            })
        );
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({ "type": "json_object" }))
        );
    }

    #[test]
    fn openai_compatible_mistral_runtime_provider_normalizes_chat_options() {
        use crate::core::ProviderSpec;
        use crate::types::{Tool, ToolChoice};

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "mistral".to_string(),
                name: "Mistral AI".to_string(),
                base_url: "https://api.mistral.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "mistral".to_string(),
            "https://api.mistral.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("mistral-small-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "lookup",
                "Lookup value",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::None)
            .build()
            .with_provider_option(
                "mistral",
                serde_json::json!({
                    "safe_prompt": false,
                    "safePrompt": true,
                    "document_image_limit": 4,
                    "documentImageLimit": 8,
                    "document_page_limit": 12,
                    "documentPageLimit": 16,
                    "structuredOutputs": false,
                    "strictJsonSchema": false,
                    "parallelToolCalls": false,
                    "reasoningEffort": "none"
                }),
            )
            .with_response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            ));

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["safe_prompt"], serde_json::json!(true));
        assert_eq!(body["document_image_limit"], serde_json::json!(8));
        assert_eq!(body["document_page_limit"], serde_json::json!(16));
        assert_eq!(body["reasoning_effort"], serde_json::json!("none"));
        assert_eq!(body["parallel_tool_calls"], serde_json::json!(false));
        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({ "type": "json_object" }))
        );
        assert!(body.get("safePrompt").is_none());
        assert!(body.get("documentImageLimit").is_none());
        assert!(body.get("documentPageLimit").is_none());
        assert!(body.get("parallelToolCalls").is_none());
        assert!(body.get("structuredOutputs").is_none());
        assert!(body.get("strictJsonSchema").is_none());
    }

    #[test]
    fn openai_compatible_mistral_tool_choice_required_maps_to_any() {
        use crate::core::ProviderSpec;
        use crate::types::{Tool, ToolChoice};

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "mistral".to_string(),
                name: "Mistral AI".to_string(),
                base_url: "https://api.mistral.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "mistral".to_string(),
            "https://api.mistral.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("mistral-large-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![Tool::function(
                "lookup",
                "Lookup value",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .tool_choice(ToolChoice::Required)
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["tool_choice"], serde_json::json!("any"));
    }

    #[test]
    fn openai_compatible_mistral_tool_choice_tool_filters_tools_and_maps_to_any() {
        use crate::core::ProviderSpec;
        use crate::types::{Tool, ToolChoice};

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "mistral".to_string(),
                name: "Mistral AI".to_string(),
                base_url: "https://api.mistral.ai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "mistral".to_string(),
            "https://api.mistral.ai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("mistral-large-latest")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![
                Tool::function(
                    "lookup",
                    "Lookup value",
                    serde_json::json!({ "type": "object", "properties": {} }),
                ),
                Tool::function(
                    "weather",
                    "Get weather",
                    serde_json::json!({ "type": "object", "properties": {} }),
                ),
            ])
            .tool_choice(ToolChoice::tool("weather"))
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["tool_choice"], serde_json::json!("any"));
        let tools = body["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["function"]["name"], serde_json::json!("weather"));
    }

    #[test]
    fn openai_compatible_perplexity_runtime_provider_normalizes_chat_options() {
        use crate::core::ProviderSpec;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "perplexity".to_string(),
                name: "Perplexity".to_string(),
                base_url: "https://api.perplexity.ai".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "perplexity".to_string(),
            "https://api.perplexity.ai".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("sonar")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .build()
            .with_provider_option(
                "perplexity",
                serde_json::json!({
                    "search_mode": "web",
                    "searchMode": "academic",
                    "return_images": false,
                    "returnImages": true,
                    "searchRecencyFilter": "month",
                    "webSearchOptions": {
                        "searchContextSize": "high",
                        "userLocation": {
                            "country": "US"
                        }
                    },
                    "someVendorParam": true
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["search_mode"], serde_json::json!("academic"));
        assert_eq!(body["return_images"], serde_json::json!(true));
        assert_eq!(body["search_recency_filter"], serde_json::json!("month"));
        assert_eq!(
            body["web_search_options"]["search_context_size"],
            serde_json::json!("high")
        );
        assert_eq!(
            body["web_search_options"]["user_location"]["country"],
            serde_json::json!("US")
        );
        assert_eq!(body["someVendorParam"], serde_json::json!(true));
        assert!(body.get("searchMode").is_none());
        assert!(body.get("returnImages").is_none());
        assert!(body.get("webSearchOptions").is_none());
    }

    #[test]
    fn openai_compatible_perplexity_json_object_uses_empty_json_schema_response_format() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "perplexity".to_string(),
                name: "Perplexity".to_string(),
                base_url: "https://api.perplexity.ai".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "perplexity".to_string(),
            "https://api.perplexity.ai".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model("sonar")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_object())
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {}
            }))
        );
    }

    #[test]
    fn openai_compatible_perplexity_json_schema_preserves_stable_schema_options() {
        use crate::core::ProviderSpec;
        use crate::types::chat::ResponseFormat;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "perplexity".to_string(),
                name: "Perplexity".to_string(),
                base_url: "https://api.perplexity.ai".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "perplexity".to_string(),
            "https://api.perplexity.ai".to_string(),
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
            .model("sonar")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(
                ResponseFormat::json_schema(schema.clone())
                    .with_name("custom")
                    .with_description("custom description")
                    .with_strict(false),
            )
            .build()
            .with_provider_option(
                "perplexity",
                serde_json::json!({ "strictJsonSchema": true, "structuredOutputs": false }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(
            body.get("response_format"),
            Some(&serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "custom",
                    "description": "custom description",
                    "schema": schema,
                    "strict": false
                }
            }))
        );
    }

    #[test]
    fn openai_compatible_perplexity_chat_settings_follow_ai_sdk_body_shape() {
        use crate::core::ProviderSpec;
        use crate::types::CommonParams;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "perplexity".to_string(),
                name: "Perplexity".to_string(),
                base_url: "https://api.perplexity.ai".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "perplexity".to_string(),
            "https://api.perplexity.ai".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model_params(CommonParams {
                model: "sonar".to_string(),
                top_k: Some(7.0),
                stop_sequences: Some(vec!["END".to_string()]),
                seed: Some(42),
                ..CommonParams::default()
            })
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .build();

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body.get("top_k"), Some(&serde_json::json!(7.0)));
        assert!(body.get("seed").is_none());
        assert!(body.get("stop").is_none());
    }

    #[test]
    fn openai_compatible_perplexity_passthrough_options_can_still_set_seed_and_stop() {
        use crate::core::ProviderSpec;
        use crate::types::CommonParams;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "perplexity".to_string(),
                name: "Perplexity".to_string(),
                base_url: "https://api.perplexity.ai".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: false,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));
        let ctx = ProviderContext::new(
            "perplexity".to_string(),
            "https://api.perplexity.ai".to_string(),
            Some("k".to_string()),
            Default::default(),
        );
        let req = crate::types::ChatRequest::builder()
            .model_params(CommonParams {
                model: "sonar".to_string(),
                stop_sequences: Some(vec!["END".to_string()]),
                seed: Some(42),
                ..CommonParams::default()
            })
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .build()
            .with_provider_option(
                "perplexity",
                serde_json::json!({ "seed": 99, "stop": ["RAW"] }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body.get("seed"), Some(&serde_json::json!(99)));
        assert_eq!(body.get("stop"), Some(&serde_json::json!(["RAW"])));
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
                    "reasoningEffort": "high",
                    "enableReasoning": true,
                    "reasoningBudget": 2048
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert!(body.get("stop").is_none());
        assert_eq!(body["reasoning_effort"], serde_json::json!("high"));
        assert!(body.get("enableReasoning").is_none());
        assert!(body.get("enable_reasoning").is_none());
        assert!(body.get("reasoningBudget").is_none());
        assert!(body.get("reasoning_budget").is_none());
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
    fn openai_compatible_xai_chat_settings_strip_unsupported_standard_fields() {
        use crate::core::ProviderSpec;
        use crate::types::{CommonParams, chat::ResponseFormat};

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
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "string" } },
            "required": ["answer"],
            "additionalProperties": false
        });
        let req = crate::types::ChatRequest::builder()
            .model_params(CommonParams {
                model: "grok-4".to_string(),
                frequency_penalty: Some(0.2),
                presence_penalty: Some(0.4),
                stop_sequences: Some(vec!["END".to_string()]),
                seed: Some(42),
                ..CommonParams::default()
            })
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .response_format(ResponseFormat::json_schema(schema.clone()).with_name("response"))
            .stream(true)
            .build()
            .with_provider_option("xai", serde_json::json!({ "strictJsonSchema": false }));

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
        assert!(body.get("stop").is_none());
        assert!(body.get("stream_options").is_none());
        assert_eq!(body.get("seed"), Some(&serde_json::json!(42)));
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

    #[test]
    fn openai_compatible_groq_runtime_normalizes_passthrough_option_keys() {
        use crate::core::ProviderSpec;

        let spec = OpenAiCompatibleSpecWithAdapter::new(Arc::new(ConfigurableAdapter::new(
            ProviderConfig {
                id: "groq".to_string(),
                name: "Groq".to_string(),
                base_url: "https://api.groq.com/openai/v1".to_string(),
                field_mappings: Default::default(),
                capabilities: vec!["tools".into()],
                default_model: None,
                supports_reasoning: true,
                api_key_env: None,
                api_key_env_aliases: vec![],
            },
        )));

        let ctx = ProviderContext::new(
            "groq".to_string(),
            "https://api.groq.com/openai/v1".to_string(),
            Some("k".to_string()),
            Default::default(),
        );

        let req = crate::types::ChatRequest::builder()
            .model("llama-3.3-70b-versatile")
            .messages(vec![crate::types::ChatMessage::user("hi").build()])
            .tools(vec![crate::types::Tool::function(
                "get_weather",
                "Get weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            )])
            .response_format(crate::types::chat::ResponseFormat::json_schema(
                serde_json::json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } }
                }),
            ))
            .build()
            .with_provider_option(
                "groq",
                serde_json::json!({
                    "serviceTier": "performance",
                    "reasoningFormat": "parsed",
                    "topLogprobs": 2,
                    "parallelToolCalls": false,
                    "strictJsonSchema": false
                }),
            );

        let bundle = spec.choose_chat_transformers(&req, &ctx);
        let body = bundle.request.transform_chat(&req).expect("transform");
        let hook = spec.chat_before_send(&req, &ctx).expect("before_send");
        let body = hook(&body).expect("hook body");

        assert_eq!(body["service_tier"], serde_json::json!("performance"));
        assert_eq!(body["reasoning_format"], serde_json::json!("parsed"));
        assert_eq!(body["top_logprobs"], serde_json::json!(2));
        assert_eq!(body["parallel_tool_calls"], serde_json::json!(false));
        assert_eq!(
            body["response_format"]["json_schema"]["strict"],
            serde_json::json!(false)
        );
        assert!(body.get("serviceTier").is_none());
        assert!(body.get("reasoningFormat").is_none());
        assert!(body.get("topLogprobs").is_none());
        assert!(body.get("strictJsonSchema").is_none());
    }
}
