//! Amazon Bedrock Chat Standard (Vercel-aligned).
//!
//! Vercel reference: `repo-ref/ai/packages/amazon-bedrock/src/bedrock-chat-language-model.ts`

use crate::core::{ChatTransformers, ProviderContext, ProviderSpec};
use crate::error::LlmError;
use crate::execution::transformers::request::RequestTransformer;
use crate::execution::transformers::response::ResponseTransformer;
use crate::provider_options::{
    BedrockCachePointConfig, BedrockChatOptions, BedrockFilePartProviderOptions,
    BedrockReasoningType, BedrockServiceTier,
};
use crate::types::{
    ChatMessage, ChatRequest, ChatResponse, ContentPart, FilePartSource, FinishReason,
    MessageContent, ProviderOptionsMap, ResponseFormat, Tool, ToolChoice, ToolResultContentPart,
    ToolResultOutput, Usage, Warning,
};
use reqwest::header::HeaderMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

mod streaming;
pub use streaming::BedrockEventConverter;

#[derive(Clone, Default)]
pub struct BedrockChatStandard;

impl BedrockChatStandard {
    pub fn new() -> Self {
        Self
    }

    pub fn create_spec(&self, provider_id: &'static str) -> BedrockChatSpec {
        BedrockChatSpec { provider_id }
    }

    pub fn create_transformers(
        &self,
        provider_id: &str,
        uses_json_response_tool: bool,
        default_model: Option<String>,
        warnings: Vec<Warning>,
        include_raw_chunks: bool,
    ) -> ChatTransformers {
        ChatTransformers {
            request: Arc::new(BedrockChatRequestTransformer {
                provider_id: provider_id.to_string(),
            }),
            response: Arc::new(BedrockChatResponseTransformer {
                provider_id: provider_id.to_string(),
                uses_json_response_tool,
                default_model: default_model.clone(),
                warnings: warnings.clone(),
            }),
            stream: None,
            json: Some(Arc::new(BedrockEventConverter::new(
                provider_id,
                uses_json_response_tool,
                default_model,
                warnings,
                include_raw_chunks,
            ))),
        }
    }
}

/// ProviderSpec implementation for Amazon Bedrock Converse.
pub struct BedrockChatSpec {
    provider_id: &'static str,
}

impl ProviderSpec for BedrockChatSpec {
    fn id(&self) -> &'static str {
        self.provider_id
    }

    fn capabilities(&self) -> crate::traits::ProviderCapabilities {
        crate::traits::ProviderCapabilities::new()
            .with_chat()
            .with_streaming()
            .with_tools()
    }

    fn build_headers(&self, ctx: &ProviderContext) -> Result<HeaderMap, LlmError> {
        crate::standards::bedrock::headers::build_bedrock_json_headers(ctx)
    }

    fn classify_http_error(
        &self,
        status: u16,
        body_text: &str,
        headers: &HeaderMap,
    ) -> Option<LlmError> {
        crate::standards::bedrock::errors::classify_bedrock_http_error(
            self.provider_id,
            status,
            body_text,
            headers,
        )
    }

    fn try_chat_url(
        &self,
        stream: bool,
        req: &ChatRequest,
        ctx: &ProviderContext,
    ) -> Result<String, LlmError> {
        let model = urlencoding::encode(&req.common_params.model);
        let suffix = if stream {
            "converse-stream"
        } else {
            "converse"
        };
        Ok(crate::utils::url::join_url(
            &ctx.base_url,
            &format!("/model/{model}/{suffix}"),
        ))
    }

    fn choose_chat_transformers(
        &self,
        req: &ChatRequest,
        _ctx: &ProviderContext,
    ) -> ChatTransformers {
        let (uses_json_response_tool, warnings) =
            match BedrockChatRequestTransformer::build_request_plan(req) {
                Ok(plan) => (plan.uses_json_response_tool, plan.warnings),
                Err(_) => {
                    let uses_json_response_tool = matches!(
                        req.response_format.as_ref(),
                        Some(ResponseFormat::Json { .. })
                    );
                    let (_, warnings) = BedrockChatRequestTransformer::build_tool_config(
                        req,
                        uses_json_response_tool,
                    );
                    (uses_json_response_tool, warnings)
                }
            };
        let default_model =
            (!req.common_params.model.trim().is_empty()).then_some(req.common_params.model.clone());

        BedrockChatStandard::new().create_transformers(
            self.provider_id,
            uses_json_response_tool,
            default_model,
            warnings,
            req.stream_options.include_raw_chunks,
        )
    }
}

struct BedrockChatRequestTransformer {
    #[allow(dead_code)]
    provider_id: String,
}

struct BedrockRequestPlan {
    uses_json_response_tool: bool,
    warnings: Vec<Warning>,
    tool_config: Option<serde_json::Value>,
    inference_config: serde_json::Map<String, serde_json::Value>,
    additional_model_request_fields: Option<serde_json::Map<String, serde_json::Value>>,
    service_tier: Option<BedrockServiceTier>,
    extra_body_fields: serde_json::Map<String, serde_json::Value>,
    additional_model_response_field_paths: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct BedrockReasoningPartOptions {
    #[serde(default)]
    signature: Option<String>,
    #[serde(rename = "redactedData", alias = "redacted_data", default)]
    redacted_data: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct BedrockMessageProviderOptions {
    #[serde(rename = "cachePoint", alias = "cache_point", default)]
    cache_point: Option<BedrockCachePointConfig>,
    #[serde(flatten, default)]
    _extra_params: HashMap<String, serde_json::Value>,
}

impl BedrockChatRequestTransformer {
    fn response_format_schema(req: &ChatRequest) -> Option<serde_json::Value> {
        match req.response_format.as_ref() {
            Some(ResponseFormat::Json { schema, .. }) => Some(schema.clone()),
            _ => None,
        }
    }

    fn build_tool_config(
        req: &ChatRequest,
        uses_json_tool: bool,
    ) -> (Option<serde_json::Value>, Vec<Warning>) {
        let mut warnings: Vec<Warning> = Vec::new();
        let mut tools: Vec<Tool> = req.tools.clone().unwrap_or_default();

        if uses_json_tool {
            let schema = Self::response_format_schema(req)
                .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
            tools.push(Tool::function(
                "json",
                "Respond with a JSON object.",
                schema,
            ));
        }

        if tools.is_empty() {
            return (None, warnings);
        }

        let tool_choice = if uses_json_tool {
            Some(ToolChoice::Required)
        } else {
            req.tool_choice.clone()
        };

        if matches!(tool_choice, Some(ToolChoice::None)) {
            return (None, warnings);
        }

        let mut bedrock_tools: Vec<serde_json::Value> = Vec::new();
        for t in tools {
            match t {
                Tool::Function { function } => {
                    let mut tool_spec = serde_json::json!({
                        "name": function.name,
                        "inputSchema": { "json": function.parameters },
                    });
                    if !function.description.trim().is_empty() {
                        tool_spec["description"] = serde_json::Value::String(function.description);
                    }
                    bedrock_tools.push(serde_json::json!({ "toolSpec": tool_spec }));
                }
                Tool::ProviderDefined(tool) => {
                    warnings.push(Warning::unsupported_tool(
                        tool.id,
                        Some("provider-defined tools are not yet supported for Amazon Bedrock"),
                    ));
                }
            }
        }

        if bedrock_tools.is_empty() {
            return (None, warnings);
        }

        let mut tool_config = serde_json::json!({ "tools": bedrock_tools });

        if let Some(tc) = tool_choice {
            let mapped = match tc {
                ToolChoice::Auto => serde_json::json!({ "auto": {} }),
                ToolChoice::Required => serde_json::json!({ "any": {} }),
                ToolChoice::Tool { name } => serde_json::json!({ "tool": { "name": name } }),
                ToolChoice::None => serde_json::Value::Null,
            };
            if !mapped.is_null() {
                tool_config["toolChoice"] = mapped;
            }
        }

        (Some(tool_config), warnings)
    }

    fn parse_bedrock_chat_options(req: &ChatRequest) -> Result<BedrockChatOptions, LlmError> {
        let Some(raw_options) = req.provider_options_map.get("bedrock") else {
            return Ok(BedrockChatOptions::default());
        };

        serde_json::from_value(raw_options.clone()).map_err(|error| {
            LlmError::InvalidParameter(format!("providerOptions.bedrock is invalid: {error}"))
        })
    }

    fn additional_model_request_fields(
        value: Option<serde_json::Value>,
    ) -> Result<serde_json::Map<String, serde_json::Value>, LlmError> {
        match value {
            Some(serde_json::Value::Object(map)) => Ok(map),
            Some(_) => Err(LlmError::InvalidParameter(
                "providerOptions.bedrock.additionalModelRequestFields must be a JSON object"
                    .to_string(),
            )),
            None => Ok(serde_json::Map::new()),
        }
    }

    fn merge_nested_object(
        target: &mut serde_json::Map<String, serde_json::Value>,
        key: &str,
        mut incoming: serde_json::Map<String, serde_json::Value>,
    ) {
        match target.get_mut(key) {
            Some(serde_json::Value::Object(existing)) => {
                existing.append(&mut incoming);
            }
            _ => {
                target.insert(key.to_string(), serde_json::Value::Object(incoming));
            }
        }
    }

    fn is_anthropic_model(model: &str) -> bool {
        model.contains("anthropic")
    }

    fn is_openai_model(model: &str) -> bool {
        model.starts_with("openai.")
    }

    fn bedrock_model_supports_native_structured_output(model: &str) -> bool {
        model.contains("claude-sonnet-4-6")
            || model.contains("claude-opus-4-6")
            || model.contains("claude-sonnet-4-5")
            || model.contains("claude-opus-4-5")
            || model.contains("claude-haiku-4-5")
            || model.contains("claude-opus-4-1")
    }

    fn build_request_plan(req: &ChatRequest) -> Result<BedrockRequestPlan, LlmError> {
        let mut warnings = Vec::new();
        let bedrock_options = Self::parse_bedrock_chat_options(req)?;
        let model = req.common_params.model.as_str();
        let is_anthropic_model = Self::is_anthropic_model(model);
        let is_openai_model = Self::is_openai_model(model);

        if req.common_params.frequency_penalty.is_some() {
            warnings.push(Warning::unsupported("frequencyPenalty", None::<String>));
        }
        if req.common_params.presence_penalty.is_some() {
            warnings.push(Warning::unsupported("presencePenalty", None::<String>));
        }
        if req.common_params.seed.is_some() {
            warnings.push(Warning::unsupported("seed", None::<String>));
        }

        let mut max_tokens = req
            .common_params
            .max_completion_tokens
            .or(req.common_params.max_tokens);
        let mut temperature = req.common_params.temperature;
        if let Some(value) = temperature {
            if value > 1.0 {
                warnings.push(Warning::unsupported(
                    "temperature",
                    Some(format!(
                        "{value} exceeds bedrock maximum of 1.0. clamped to 1.0"
                    )),
                ));
                temperature = Some(1.0);
            } else if value < 0.0 {
                warnings.push(Warning::unsupported(
                    "temperature",
                    Some(format!(
                        "{value} is below bedrock minimum of 0. clamped to 0"
                    )),
                ));
                temperature = Some(0.0);
            }
        }

        let mut additional_model_request_fields = Self::additional_model_request_fields(
            bedrock_options.additional_model_request_fields.clone(),
        )?;

        let reasoning_config = bedrock_options.reasoning_config.as_ref();
        let thinking_type = reasoning_config.and_then(|config| config.r#type);
        let thinking_budget = if matches!(thinking_type, Some(BedrockReasoningType::Enabled)) {
            reasoning_config.and_then(|config| config.budget_tokens)
        } else {
            None
        };
        let is_anthropic_thinking_enabled = is_anthropic_model
            && matches!(
                thinking_type,
                Some(BedrockReasoningType::Enabled | BedrockReasoningType::Adaptive)
            );

        if is_anthropic_model {
            match thinking_type {
                Some(BedrockReasoningType::Enabled) => {
                    if let Some(budget_tokens) = thinking_budget {
                        max_tokens = Some(max_tokens.unwrap_or(4096).saturating_add(budget_tokens));
                        additional_model_request_fields.insert(
                            "thinking".to_string(),
                            serde_json::json!({
                                "type": "enabled",
                                "budget_tokens": budget_tokens,
                            }),
                        );
                    }
                }
                Some(BedrockReasoningType::Adaptive) => {
                    additional_model_request_fields.insert(
                        "thinking".to_string(),
                        serde_json::json!({ "type": "adaptive" }),
                    );
                }
                _ => {}
            }
        } else if let Some(reasoning_config) = reasoning_config {
            if reasoning_config.budget_tokens.is_some() {
                warnings.push(Warning::unsupported(
                    "budgetTokens",
                    Some(
                        "budgetTokens applies only to Anthropic models on Bedrock and will be ignored for this model.",
                    ),
                ));
            }
            if matches!(
                reasoning_config.r#type,
                Some(BedrockReasoningType::Adaptive)
            ) {
                warnings.push(Warning::unsupported(
                    "adaptive thinking",
                    Some("adaptive thinking type applies only to Anthropic models on Bedrock."),
                ));
            }
        }

        if let Some(max_reasoning_effort) =
            reasoning_config.and_then(|config| config.max_reasoning_effort)
        {
            if is_anthropic_model {
                let mut output_config = serde_json::Map::new();
                output_config.insert(
                    "effort".to_string(),
                    serde_json::json!(max_reasoning_effort),
                );
                Self::merge_nested_object(
                    &mut additional_model_request_fields,
                    "output_config",
                    output_config,
                );
            } else if is_openai_model {
                additional_model_request_fields.insert(
                    "reasoning_effort".to_string(),
                    serde_json::json!(max_reasoning_effort),
                );
            } else {
                let mut nested_reasoning_config = serde_json::Map::new();
                if let Some(reasoning_type) = reasoning_config.and_then(|config| config.r#type)
                    && !matches!(reasoning_type, BedrockReasoningType::Adaptive)
                {
                    nested_reasoning_config
                        .insert("type".to_string(), serde_json::json!(reasoning_type));
                }
                if let Some(budget_tokens) = thinking_budget {
                    nested_reasoning_config
                        .insert("budgetTokens".to_string(), serde_json::json!(budget_tokens));
                }
                nested_reasoning_config.insert(
                    "maxReasoningEffort".to_string(),
                    serde_json::json!(max_reasoning_effort),
                );
                additional_model_request_fields.insert(
                    "reasoningConfig".to_string(),
                    serde_json::Value::Object(nested_reasoning_config),
                );
            }
        }

        if let Some(anthropic_beta) = bedrock_options.anthropic_beta.clone()
            && !anthropic_beta.is_empty()
        {
            additional_model_request_fields.insert(
                "anthropic_beta".to_string(),
                serde_json::json!(anthropic_beta),
            );
        }

        let response_schema = Self::response_format_schema(req);
        let use_native_structured_output = is_anthropic_model
            && response_schema.is_some()
            && (Self::bedrock_model_supports_native_structured_output(model)
                || is_anthropic_thinking_enabled);
        if let Some(schema) = response_schema.clone()
            && use_native_structured_output
        {
            let mut output_config = serde_json::Map::new();
            output_config.insert(
                "format".to_string(),
                serde_json::json!({
                    "type": "json_schema",
                    "schema": schema,
                }),
            );
            Self::merge_nested_object(
                &mut additional_model_request_fields,
                "output_config",
                output_config,
            );
        }

        let mut top_p = req.common_params.top_p;
        let mut top_k = req.common_params.top_k;
        if is_anthropic_thinking_enabled {
            if temperature.is_some() {
                temperature = None;
                warnings.push(Warning::unsupported(
                    "temperature",
                    Some("temperature is not supported when thinking is enabled"),
                ));
            }
            if top_p.is_some() {
                top_p = None;
                warnings.push(Warning::unsupported(
                    "topP",
                    Some("topP is not supported when thinking is enabled"),
                ));
            }
            if top_k.is_some() {
                top_k = None;
                warnings.push(Warning::unsupported(
                    "topK",
                    Some("topK is not supported when thinking is enabled"),
                ));
            }
        }

        let uses_json_response_tool = response_schema.is_some() && !use_native_structured_output;
        let (tool_config, mut tool_warnings) =
            Self::build_tool_config(req, uses_json_response_tool);
        warnings.append(&mut tool_warnings);

        let mut inference_config = serde_json::Map::new();
        if let Some(value) = max_tokens {
            inference_config.insert("maxTokens".to_string(), serde_json::json!(value));
        }
        if let Some(value) = temperature {
            inference_config.insert("temperature".to_string(), serde_json::json!(value));
        }
        if let Some(value) = top_p {
            inference_config.insert("topP".to_string(), serde_json::json!(value));
        }
        if let Some(value) = top_k {
            inference_config.insert("topK".to_string(), serde_json::json!(value));
        }
        if let Some(value) = req.common_params.stop_sequences.as_ref() {
            inference_config.insert("stopSequences".to_string(), serde_json::json!(value));
        }

        Ok(BedrockRequestPlan {
            uses_json_response_tool,
            warnings,
            tool_config,
            inference_config,
            additional_model_request_fields: (!additional_model_request_fields.is_empty())
                .then_some(additional_model_request_fields),
            service_tier: bedrock_options.service_tier,
            extra_body_fields: bedrock_options.extra_params.into_iter().collect(),
            additional_model_response_field_paths: is_anthropic_model
                .then_some(serde_json::json!(["/delta/stop_sequence"])),
        })
    }

    fn parse_provider_options<T: DeserializeOwned>(
        provider_options: &ProviderOptionsMap,
        scope: &str,
    ) -> Result<Option<T>, LlmError> {
        let Some(raw_options) = provider_options.get("bedrock") else {
            return Ok(None);
        };

        serde_json::from_value(raw_options.clone())
            .map(Some)
            .map_err(|error| {
                LlmError::InvalidParameter(format!(
                    "{scope} providerOptions.bedrock is invalid: {error}"
                ))
            })
    }

    fn cache_point(
        provider_options: &ProviderOptionsMap,
        scope: &str,
    ) -> Result<Option<serde_json::Value>, LlmError> {
        let Some(options) =
            Self::parse_provider_options::<BedrockMessageProviderOptions>(provider_options, scope)?
        else {
            return Ok(None);
        };

        Ok(options
            .cache_point
            .map(|cache_point| serde_json::json!({ "cachePoint": cache_point })))
    }

    fn base64_source(source: &FilePartSource, label: &str) -> Result<String, LlmError> {
        match source {
            FilePartSource::Media(source) => source.as_base64().ok_or_else(|| {
                LlmError::UnsupportedOperation(format!(
                    "Bedrock chat request does not support {label} with URL sources"
                ))
            }),
            FilePartSource::ProviderReference { .. } => Err(LlmError::UnsupportedOperation(
                format!("Bedrock chat request does not support {label} with provider references"),
            )),
        }
    }

    fn strip_file_extension(filename: &str) -> String {
        match filename.find('.') {
            Some(index) => filename[..index].to_string(),
            None => filename.to_string(),
        }
    }

    fn next_document_name(counter: &mut usize) -> String {
        *counter += 1;
        format!("document-{}", *counter)
    }

    fn trim_if_last(
        is_last_block: bool,
        is_last_message: bool,
        is_last_content_part: bool,
        text: &str,
    ) -> String {
        if is_last_block && is_last_message && is_last_content_part {
            text.trim().to_string()
        } else {
            text.to_string()
        }
    }

    fn bedrock_image_format(mime_type: &str) -> Result<&'static str, LlmError> {
        match mime_type {
            "image/jpeg" => Ok("jpeg"),
            "image/png" => Ok("png"),
            "image/gif" => Ok("gif"),
            "image/webp" => Ok("webp"),
            _ => Err(LlmError::UnsupportedOperation(format!(
                "Unsupported image mime type: {mime_type}, expected one of: image/jpeg, image/png, image/gif, image/webp"
            ))),
        }
    }

    fn bedrock_document_format(mime_type: &str) -> Result<&'static str, LlmError> {
        match mime_type {
            "application/pdf" => Ok("pdf"),
            "text/csv" => Ok("csv"),
            "application/msword" => Ok("doc"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Ok("docx"),
            "application/vnd.ms-excel" => Ok("xls"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Ok("xlsx"),
            "text/html" => Ok("html"),
            "text/plain" => Ok("txt"),
            "text/markdown" => Ok("md"),
            _ => Err(LlmError::UnsupportedOperation(format!(
                "Unsupported file mime type: {mime_type}, expected one of: application/pdf, text/csv, application/msword, application/vnd.openxmlformats-officedocument.wordprocessingml.document, application/vnd.ms-excel, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, text/html, text/plain, text/markdown"
            ))),
        }
    }

    fn tool_result_content_blocks(
        output: &ToolResultOutput,
    ) -> Result<Vec<serde_json::Value>, LlmError> {
        match output {
            ToolResultOutput::Content { value, .. } => value
                .iter()
                .map(|content_part| match content_part {
                    ToolResultContentPart::Text { text, .. } => {
                        Ok(serde_json::json!({ "text": text }))
                    }
                    ToolResultContentPart::ImageData {
                        data, media_type, ..
                    } => {
                        if !media_type.starts_with("image/") {
                            return Err(LlmError::UnsupportedOperation(format!(
                                "Unsupported tool result image media type: {media_type}"
                            )));
                        }

                        let format = Self::bedrock_image_format(media_type)?;
                        Ok(serde_json::json!({
                            "image": {
                                "format": format,
                                "source": { "bytes": data },
                            }
                        }))
                    }
                    ToolResultContentPart::FileData { .. } => Err(LlmError::UnsupportedOperation(
                        "Bedrock chat request does not support tool result content part type `file-data`"
                            .to_string(),
                    )),
                    ToolResultContentPart::FileUrl { .. } => Err(LlmError::UnsupportedOperation(
                        "Bedrock chat request does not support tool result content part type `file-url`"
                            .to_string(),
                    )),
                    ToolResultContentPart::FileId { .. } => Err(LlmError::UnsupportedOperation(
                        "Bedrock chat request does not support tool result content part type `file-id`"
                            .to_string(),
                    )),
                    ToolResultContentPart::FileReference { .. } => Err(
                        LlmError::UnsupportedOperation(
                            "Bedrock chat request does not support tool result content part type `file-reference`"
                                .to_string(),
                        ),
                    ),
                    ToolResultContentPart::ImageUrl { .. } => Err(
                        LlmError::UnsupportedOperation(
                            "Bedrock chat request does not support tool result content part type `image-url`"
                                .to_string(),
                        ),
                    ),
                    ToolResultContentPart::ImageFileId { .. } => Err(
                        LlmError::UnsupportedOperation(
                            "Bedrock chat request does not support tool result content part type `image-file-id`"
                                .to_string(),
                        ),
                    ),
                    ToolResultContentPart::ImageFileReference { .. } => Err(
                        LlmError::UnsupportedOperation(
                            "Bedrock chat request does not support tool result content part type `image-file-reference`"
                                .to_string(),
                        ),
                    ),
                    ToolResultContentPart::Custom { .. } => Err(LlmError::UnsupportedOperation(
                        "Bedrock chat request does not support tool result content part type `custom`"
                            .to_string(),
                    )),
                })
                .collect(),
            ToolResultOutput::Text { value, .. } | ToolResultOutput::ErrorText { value, .. } => {
                Ok(vec![serde_json::json!({ "text": value })])
            }
            ToolResultOutput::ExecutionDenied { reason, .. } => Ok(vec![serde_json::json!({
                "text": reason.clone().unwrap_or_else(|| "Tool call execution denied.".to_string())
            })]),
            ToolResultOutput::Json { value, .. } | ToolResultOutput::ErrorJson { value, .. } => {
                Ok(vec![serde_json::json!({ "text": value.to_string() })])
            }
        }
    }

    fn split_system_messages(
        messages: &[ChatMessage],
    ) -> Result<(Vec<serde_json::Value>, &[ChatMessage]), LlmError> {
        let mut system: Vec<serde_json::Value> = Vec::new();
        let mut idx = 0usize;
        while idx < messages.len() {
            match messages[idx].role {
                crate::types::MessageRole::System | crate::types::MessageRole::Developer => {
                    let text = messages[idx].content.all_text();
                    system.push(serde_json::json!({ "text": text }));
                    if let Some(cache_point) =
                        Self::cache_point(&messages[idx].provider_options, "ChatMessage")?
                    {
                        system.push(cache_point);
                    }
                    idx += 1;
                }
                _ => break,
            }
        }
        Ok((system, &messages[idx..]))
    }

    fn content_parts(message: &ChatMessage) -> Vec<ContentPart> {
        #[allow(unreachable_patterns)]
        match &message.content {
            MessageContent::Text(t) => vec![ContentPart::text(t.clone())],
            MessageContent::MultiModal(parts) => parts.clone(),
            _ => vec![ContentPart::text(message.content.all_text())],
        }
    }

    fn convert_messages(
        messages: &[ChatMessage],
        is_mistral: bool,
    ) -> Result<Vec<serde_json::Value>, LlmError> {
        let mut out: Vec<serde_json::Value> = Vec::new();
        let mut i = 0usize;
        let mut document_counter = 0usize;

        while i < messages.len() {
            match messages[i].role {
                crate::types::MessageRole::User | crate::types::MessageRole::Tool => {
                    let mut content: Vec<serde_json::Value> = Vec::new();
                    while i < messages.len()
                        && matches!(
                            messages[i].role,
                            crate::types::MessageRole::User | crate::types::MessageRole::Tool
                        )
                    {
                        let msg = &messages[i];
                        match msg.role {
                            crate::types::MessageRole::User => {
                                for part in Self::content_parts(msg) {
                                    match part {
                                        ContentPart::Text { text, .. } => {
                                            content.push(serde_json::json!({ "text": text }));
                                        }
                                        ContentPart::File {
                                            source,
                                            media_type,
                                            filename,
                                            provider_options,
                                            ..
                                        } => {
                                            let bytes = Self::base64_source(&source, "file parts")?;

                                            if media_type.starts_with("image/") {
                                                let format =
                                                    Self::bedrock_image_format(&media_type)?;
                                                content.push(serde_json::json!({
                                                    "image": {
                                                        "format": format,
                                                        "source": { "bytes": bytes },
                                                    }
                                                }));
                                                continue;
                                            }

                                            let format =
                                                Self::bedrock_document_format(&media_type)?;
                                            let file_options = Self::parse_provider_options::<
                                                BedrockFilePartProviderOptions,
                                            >(
                                                &provider_options,
                                                "ContentPart::File",
                                            )?
                                            .unwrap_or_default();
                                            let name = filename
                                                .as_deref()
                                                .map(Self::strip_file_extension)
                                                .unwrap_or_else(|| {
                                                    Self::next_document_name(&mut document_counter)
                                                });

                                            let mut document = serde_json::Map::new();
                                            document.insert(
                                                "format".to_string(),
                                                serde_json::Value::String(format.to_string()),
                                            );
                                            document.insert(
                                                "name".to_string(),
                                                serde_json::Value::String(name),
                                            );
                                            document.insert(
                                                "source".to_string(),
                                                serde_json::json!({ "bytes": bytes }),
                                            );
                                            if file_options
                                                .citations
                                                .map(|citations| citations.enabled)
                                                == Some(true)
                                            {
                                                document.insert(
                                                    "citations".to_string(),
                                                    serde_json::json!({ "enabled": true }),
                                                );
                                            }

                                            content.push(serde_json::json!({
                                                "document": serde_json::Value::Object(document),
                                            }));
                                        }
                                        ContentPart::Image { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support generic `image` parts. Use `file` parts with an `image/*` media_type instead."
                                                    .to_string(),
                                            ));
                                        }
                                        ContentPart::Audio { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support user audio parts"
                                                    .to_string(),
                                            ));
                                        }
                                        ContentPart::ReasoningFile { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support user reasoning-file parts"
                                                    .to_string(),
                                            ));
                                        }
                                        ContentPart::Custom { kind, .. } => {
                                            return Err(LlmError::UnsupportedOperation(format!(
                                                "Bedrock chat request does not support user custom content part `{kind}`"
                                            )));
                                        }
                                        ContentPart::Source { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support user source parts"
                                                    .to_string(),
                                            ));
                                        }
                                        ContentPart::ToolCall { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support tool-call parts inside user messages"
                                                    .to_string(),
                                            ));
                                        }
                                        ContentPart::ToolResult { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support tool-result parts inside user messages"
                                                    .to_string(),
                                            ));
                                        }
                                        ContentPart::Reasoning { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support reasoning parts inside user messages"
                                                    .to_string(),
                                            ));
                                        }
                                        ContentPart::ToolApprovalResponse { .. }
                                        | ContentPart::ToolApprovalRequest { .. } => {
                                            return Err(LlmError::UnsupportedOperation(
                                                "Bedrock chat request does not support tool approval parts inside user messages"
                                                    .to_string(),
                                            ));
                                        }
                                    }
                                }
                            }
                            crate::types::MessageRole::Tool => {
                                for part in Self::content_parts(msg) {
                                    if matches!(part, ContentPart::ToolApprovalResponse { .. }) {
                                        continue;
                                    }
                                    if let Some(tr) = part.as_tool_result() {
                                        let content_blocks =
                                            Self::tool_result_content_blocks(tr.output)?;
                                        content.push(serde_json::json!({
                                            "toolResult": {
                                                "toolUseId": normalize_tool_call_id(
                                                    tr.tool_call_id,
                                                    is_mistral,
                                                ),
                                                "content": content_blocks,
                                            }
                                        }));
                                    } else {
                                        return Err(LlmError::UnsupportedOperation(
                                            "Bedrock chat request only supports tool-result parts inside tool messages"
                                                .to_string(),
                                        ));
                                    }
                                }
                            }
                            _ => {}
                        }
                        if let Some(cache_point) =
                            Self::cache_point(&msg.provider_options, "ChatMessage")?
                        {
                            content.push(cache_point);
                        }
                        i += 1;
                    }
                    out.push(serde_json::json!({ "role": "user", "content": content }));
                }
                crate::types::MessageRole::Assistant => {
                    let mut content: Vec<serde_json::Value> = Vec::new();
                    while i < messages.len()
                        && matches!(messages[i].role, crate::types::MessageRole::Assistant)
                    {
                        let msg = &messages[i];
                        let parts = Self::content_parts(msg);
                        let parts_len = parts.len();
                        let is_last_message = i == messages.len() - 1
                            || !matches!(
                                messages[i + 1].role,
                                crate::types::MessageRole::Assistant
                            );
                        let has_reasoning_blocks = parts
                            .iter()
                            .any(|part| matches!(part, ContentPart::Reasoning { .. }));

                        for (part_index, part) in parts.into_iter().enumerate() {
                            let is_last_content_part = part_index + 1 == parts_len;
                            match part {
                                ContentPart::Text { text, .. } => {
                                    if text.trim().is_empty() && !has_reasoning_blocks {
                                        continue;
                                    }
                                    content.push(serde_json::json!({
                                        "text": Self::trim_if_last(
                                            i == messages.len() - 1,
                                            is_last_message,
                                            is_last_content_part,
                                            &text,
                                        )
                                    }));
                                }
                                ContentPart::ToolCall {
                                    tool_call_id,
                                    tool_name,
                                    arguments,
                                    ..
                                } => {
                                    content.push(serde_json::json!({
                                        "toolUse": {
                                            "toolUseId": normalize_tool_call_id(
                                                &tool_call_id,
                                                is_mistral,
                                            ),
                                            "name": tool_name,
                                            "input": arguments,
                                        }
                                    }));
                                }
                                ContentPart::Reasoning {
                                    text,
                                    provider_options,
                                    ..
                                } => {
                                    let reasoning_options = Self::parse_provider_options::<
                                        BedrockReasoningPartOptions,
                                    >(
                                        &provider_options,
                                        "ContentPart::Reasoning",
                                    )?
                                    .unwrap_or_default();

                                    if let Some(signature) = reasoning_options.signature {
                                        content.push(serde_json::json!({
                                            "reasoningContent": {
                                                "reasoningText": {
                                                    "text": text,
                                                    "signature": signature,
                                                }
                                            }
                                        }));
                                    } else if let Some(redacted_data) =
                                        reasoning_options.redacted_data
                                    {
                                        content.push(serde_json::json!({
                                            "reasoningContent": {
                                                "redactedReasoning": { "data": redacted_data }
                                            }
                                        }));
                                    } else {
                                        content.push(serde_json::json!({
                                            "reasoningContent": {
                                                "reasoningText": {
                                                    "text": Self::trim_if_last(
                                                        i == messages.len() - 1,
                                                        is_last_message,
                                                        is_last_content_part,
                                                        &text,
                                                    )
                                                }
                                            }
                                        }));
                                    }
                                }
                                ContentPart::File { .. }
                                | ContentPart::Image { .. }
                                | ContentPart::Audio { .. }
                                | ContentPart::ReasoningFile { .. }
                                | ContentPart::Custom { .. }
                                | ContentPart::Source { .. }
                                | ContentPart::ToolApprovalResponse { .. }
                                | ContentPart::ToolApprovalRequest { .. }
                                | ContentPart::ToolResult { .. } => {
                                    return Err(LlmError::UnsupportedOperation(
                                        "Bedrock chat request does not support this assistant content part"
                                            .to_string(),
                                    ));
                                }
                            }
                        }
                        if let Some(cache_point) =
                            Self::cache_point(&msg.provider_options, "ChatMessage")?
                        {
                            content.push(cache_point);
                        }
                        i += 1;
                    }
                    out.push(serde_json::json!({ "role": "assistant", "content": content }));
                }
                _ => {
                    return Err(LlmError::UnsupportedOperation(
                        "System/developer messages must be placed at the beginning for Bedrock"
                            .to_string(),
                    ));
                }
            }
        }

        Ok(out)
    }
}

impl RequestTransformer for BedrockChatRequestTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let plan = Self::build_request_plan(req)?;
        let (system, rest) = Self::split_system_messages(&req.messages)?;
        let messages = Self::convert_messages(
            rest,
            is_mistral_model(Some(req.common_params.model.as_str())),
        )?;

        let mut body = serde_json::json!({
            "system": system,
            "messages": messages,
        });
        if !plan.inference_config.is_empty() {
            body["inferenceConfig"] = serde_json::Value::Object(plan.inference_config);
        }
        if let Some(fields) = plan.additional_model_request_fields {
            body["additionalModelRequestFields"] = serde_json::Value::Object(fields);
        }
        if let Some(service_tier) = plan.service_tier {
            body["serviceTier"] = serde_json::json!({
                "type": service_tier,
            });
        }
        if let Some(paths) = plan.additional_model_response_field_paths {
            body["additionalModelResponseFieldPaths"] = paths;
        }
        if let Some(body_object) = body.as_object_mut() {
            for (key, value) in plan.extra_body_fields {
                body_object.insert(key, value);
            }
        }
        if let Some(cfg) = plan.tool_config {
            body["toolConfig"] = cfg;
        }

        Ok(body)
    }
}

#[derive(Clone)]
struct BedrockChatResponseTransformer {
    provider_id: String,
    uses_json_response_tool: bool,
    default_model: Option<String>,
    warnings: Vec<Warning>,
}

impl BedrockChatResponseTransformer {
    fn map_finish_reason(
        raw: Option<&str>,
        is_json_response_from_tool: bool,
    ) -> Option<FinishReason> {
        let raw = raw?;
        Some(match raw {
            "stop" | "stop_sequence" | "end_turn" => FinishReason::Stop,
            "max_tokens" | "length" => FinishReason::Length,
            "content_filtered" | "guardrail_intervened" | "content-filter" | "content_filter" => {
                FinishReason::ContentFilter
            }
            "tool_use" | "tool-calls" | "tool_calls" => {
                if is_json_response_from_tool {
                    FinishReason::Stop
                } else {
                    FinishReason::ToolCalls
                }
            }
            other => FinishReason::Other(other.to_string()),
        })
    }

    fn set_bedrock_metadata(
        resp: &mut ChatResponse,
        is_json_response_from_tool: bool,
        stop_sequence: Option<serde_json::Value>,
    ) {
        let mut bedrock = bedrock_metadata_fragment(is_json_response_from_tool, stop_sequence);
        if bedrock.is_empty() {
            return;
        }

        merge_bedrock_metadata_root(
            resp.provider_metadata.get_or_insert_with(HashMap::new),
            &mut bedrock,
        );
    }
}

fn is_mistral_model(model: Option<&str>) -> bool {
    model.is_some_and(|model| model.starts_with("mistral.") || model.contains(".mistral."))
}

fn normalize_tool_call_id(tool_call_id: &str, is_mistral: bool) -> String {
    if !is_mistral {
        return tool_call_id.to_string();
    }

    tool_call_id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(9)
        .collect()
}

fn bedrock_part_provider_metadata(
    inner: serde_json::Map<String, serde_json::Value>,
) -> Option<HashMap<String, serde_json::Value>> {
    if inner.is_empty() {
        return None;
    }

    Some(HashMap::from([(
        "bedrock".to_string(),
        serde_json::Value::Object(inner),
    )]))
}

fn bedrock_reasoning_part_metadata(
    signature: Option<&str>,
    redacted_data: Option<&str>,
) -> Option<HashMap<String, serde_json::Value>> {
    let mut bedrock = serde_json::Map::new();
    if let Some(signature) = signature {
        bedrock.insert(
            "signature".to_string(),
            serde_json::Value::String(signature.to_string()),
        );
    }
    if let Some(redacted_data) = redacted_data {
        bedrock.insert(
            "redactedData".to_string(),
            serde_json::Value::String(redacted_data.to_string()),
        );
    }
    bedrock_part_provider_metadata(bedrock)
}

fn bedrock_metadata_fragment(
    is_json_response_from_tool: bool,
    stop_sequence: Option<serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut bedrock = serde_json::Map::new();
    if is_json_response_from_tool {
        bedrock.insert(
            "isJsonResponseFromTool".to_string(),
            serde_json::Value::Bool(true),
        );
    }
    if let Some(stop_sequence) = stop_sequence {
        bedrock.insert("stopSequence".to_string(), stop_sequence);
    }
    bedrock
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BedrockUsageInfo {
    input_tokens: Option<u32>,
    output_tokens: Option<u32>,
    total_tokens: Option<u32>,
    cache_read_input_tokens: Option<u32>,
    cache_write_input_tokens: Option<u32>,
    cache_details: Option<serde_json::Value>,
}

fn bedrock_usage_metadata_fragment(
    usage: &BedrockUsageInfo,
) -> Option<(String, serde_json::Value)> {
    let mut usage_meta = serde_json::Map::new();

    if let Some(cache_write_input_tokens) = usage.cache_write_input_tokens {
        usage_meta.insert(
            "cacheWriteInputTokens".to_string(),
            serde_json::json!(cache_write_input_tokens),
        );
    }
    if let Some(cache_details) = usage.cache_details.clone() {
        usage_meta.insert("cacheDetails".to_string(), cache_details);
    }

    (!usage_meta.is_empty()).then(|| ("usage".to_string(), serde_json::Value::Object(usage_meta)))
}

fn merge_bedrock_metadata_root(
    root: &mut HashMap<String, serde_json::Value>,
    fragment: &mut serde_json::Map<String, serde_json::Value>,
) {
    if fragment.is_empty() {
        return;
    }

    match root.get_mut("bedrock") {
        Some(serde_json::Value::Object(existing)) => {
            existing.extend(std::mem::take(fragment));
        }
        _ => {
            root.insert(
                "bedrock".to_string(),
                serde_json::Value::Object(std::mem::take(fragment)),
            );
        }
    }
}

fn response_provider_metadata_to_stream_provider_metadata(
    metadata: Option<&HashMap<String, serde_json::Value>>,
) -> Option<HashMap<String, serde_json::Value>> {
    metadata.cloned()
}

fn build_bedrock_usage_from_info_with_raw(
    usage: &BedrockUsageInfo,
    raw_usage: Option<serde_json::Value>,
) -> Usage {
    let mut builder = Usage::builder();

    if let Some(input_tokens) = usage.input_tokens {
        builder = builder.prompt_tokens(input_tokens);
        builder = builder.with_input_no_cache_tokens(input_tokens);
    }
    if let Some(output_tokens) = usage.output_tokens {
        builder = builder.completion_tokens(output_tokens);
        builder = builder.with_output_total_tokens(output_tokens);
        builder = builder.with_output_text_tokens(output_tokens);
    }
    if let Some(total_tokens) = usage.total_tokens {
        builder = builder.total_tokens(total_tokens);
    }
    if let Some(cache_read_input_tokens) = usage.cache_read_input_tokens {
        builder = builder.with_input_cache_read_tokens(cache_read_input_tokens);
    }
    if let Some(cache_write_input_tokens) = usage.cache_write_input_tokens {
        builder = builder.with_input_cache_write_tokens(cache_write_input_tokens);
    }
    if let Some(input_total_tokens) = usage.input_tokens.map(|input_tokens| {
        input_tokens
            .saturating_add(usage.cache_read_input_tokens.unwrap_or(0))
            .saturating_add(usage.cache_write_input_tokens.unwrap_or(0))
    }) {
        builder = builder.with_input_total_tokens(input_total_tokens);
    }
    if let Some(raw_usage) = raw_usage {
        builder = builder.with_raw_usage_value(raw_usage);
    } else if let Ok(serde_json::Value::Object(raw)) = serde_json::to_value(usage) {
        builder = builder.with_raw_usage(raw);
    }

    builder.build()
}

fn build_bedrock_usage_from_value(value: &serde_json::Value) -> Option<Usage> {
    serde_json::from_value::<BedrockUsageInfo>(value.clone())
        .ok()
        .map(|usage| build_bedrock_usage_from_info_with_raw(&usage, Some(value.clone())))
}

impl ResponseTransformer for BedrockChatResponseTransformer {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn transform_chat_response(&self, raw: &serde_json::Value) -> Result<ChatResponse, LlmError> {
        let content_arr = raw
            .get("output")
            .and_then(|o| o.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| LlmError::ParseError("Missing Bedrock output.message.content".into()))?;

        let mut parts: Vec<ContentPart> = Vec::new();
        let mut is_json_response_from_tool = false;
        let is_mistral = is_mistral_model(self.default_model.as_deref());

        for item in content_arr {
            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                parts.push(ContentPart::text(text.to_string()));
            }

            if let Some(reasoning_content) = item.get("reasoningContent") {
                if let Some(reasoning_text) = reasoning_content
                    .get("reasoningText")
                    .and_then(|value| value.get("text"))
                    .and_then(|value| value.as_str())
                {
                    let provider_metadata = bedrock_reasoning_part_metadata(
                        reasoning_content
                            .get("reasoningText")
                            .and_then(|value| value.get("signature"))
                            .and_then(|value| value.as_str()),
                        None,
                    );
                    parts.push(ContentPart::Reasoning {
                        text: reasoning_text.to_string(),
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata,
                    });
                } else if let Some(redacted_data) = reasoning_content
                    .get("redactedReasoning")
                    .and_then(|value| value.get("data"))
                    .and_then(|value| value.as_str())
                {
                    parts.push(ContentPart::Reasoning {
                        text: String::new(),
                        provider_options: crate::types::ProviderOptionsMap::default(),
                        provider_metadata: bedrock_reasoning_part_metadata(
                            None,
                            Some(redacted_data),
                        ),
                    });
                }
            }

            if let Some(tool_use) = item.get("toolUse") {
                let raw_tool_use_id = tool_use
                    .get("toolUseId")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool-use-id");
                let name = tool_use
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tool")
                    .to_string();
                let input = tool_use
                    .get("input")
                    .cloned()
                    .unwrap_or(serde_json::json!({}));

                if self.uses_json_response_tool && name == "json" {
                    is_json_response_from_tool = true;
                    parts.push(ContentPart::text(
                        serde_json::to_string(&input).unwrap_or_default(),
                    ));
                } else {
                    parts.push(ContentPart::tool_call(
                        normalize_tool_call_id(raw_tool_use_id, is_mistral),
                        name,
                        input,
                        None,
                    ));
                }
            }
        }

        let usage = raw.get("usage").and_then(build_bedrock_usage_from_value);

        let mut resp = ChatResponse::new(MessageContent::MultiModal(parts));
        resp.model = self.default_model.clone();
        resp.usage = usage;
        if !self.warnings.is_empty() {
            resp.warnings = Some(self.warnings.clone());
        }

        let raw_reason = raw.get("stopReason").and_then(|v| v.as_str());
        resp.finish_reason = Self::map_finish_reason(raw_reason, is_json_response_from_tool);
        resp.raw_finish_reason = raw_reason.map(ToString::to_string);

        let stop_sequence = raw
            .get("additionalModelResponseFields")
            .and_then(|v| v.get("delta"))
            .and_then(|v| v.get("stop_sequence"))
            .cloned();
        Self::set_bedrock_metadata(&mut resp, is_json_response_from_tool, stop_sequence);

        let mut extra_metadata = serde_json::Map::new();
        if let Some(trace) = raw.get("trace").cloned() {
            extra_metadata.insert("trace".to_string(), trace);
        }
        if let Some(performance_config) = raw.get("performanceConfig").cloned() {
            extra_metadata.insert("performanceConfig".to_string(), performance_config);
        }
        if let Some(service_tier) = raw.get("serviceTier").cloned() {
            extra_metadata.insert("serviceTier".to_string(), service_tier);
        }
        if let Some(usage_value) = raw.get("usage")
            && let Ok(usage_info) = serde_json::from_value::<BedrockUsageInfo>(usage_value.clone())
            && let Some((key, value)) = bedrock_usage_metadata_fragment(&usage_info)
        {
            extra_metadata.insert(key, value);
        }
        if !extra_metadata.is_empty() {
            let root = resp.provider_metadata.get_or_insert_with(HashMap::new);
            merge_bedrock_metadata_root(root, &mut extra_metadata);
        }

        Ok(resp)
    }
}

#[cfg(test)]
mod tests;
