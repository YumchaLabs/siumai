use crate::error::LlmError;
use crate::execution::middleware::LanguageModelMiddleware;
use crate::providers::groq::models;
use crate::streaming::ChatStreamEvent;
use crate::types::{ChatRequest, ChatResponse, Tool, ToolChoice, Warning};

const BROWSER_SEARCH_TOOL_ID: &str = "groq.browser_search";
const BROWSER_SEARCH_TOOL_TYPE: &str = "browser_search";
const BROWSER_SEARCH_SUPPORTED_MODELS: &[&str] = &[
    models::production::GPT_OSS_20B,
    models::production::GPT_OSS_120B,
];
const STRUCTURED_OUTPUTS_WARNING_DETAILS: &str =
    "JSON response format schema is only supported with structuredOutputs";

#[derive(Debug, Default)]
pub(crate) struct GroqRequestMiddleware {
    default_structured_outputs: Option<bool>,
}

impl GroqRequestMiddleware {
    pub const fn new(default_structured_outputs: Option<bool>) -> Self {
        Self {
            default_structured_outputs,
        }
    }

    fn has_browser_search_tool(req: &ChatRequest) -> bool {
        req.tools.as_deref().unwrap_or_default().iter().any(|tool| {
            matches!(
                tool,
                Tool::ProviderDefined(provider_tool) if provider_tool.id == BROWSER_SEARCH_TOOL_ID
            )
        })
    }

    fn supports_browser_search(model: &str) -> bool {
        BROWSER_SEARCH_SUPPORTED_MODELS.contains(&model)
    }

    fn supported_models_string() -> &'static str {
        "openai/gpt-oss-20b, openai/gpt-oss-120b"
    }

    fn unsupported_warning(model: &str) -> Warning {
        Warning::unsupported(
            format!("provider-defined tool {BROWSER_SEARCH_TOOL_ID}"),
            Some(format!(
                "Browser search is only supported on the following models: {}. Current model: {}",
                Self::supported_models_string(),
                model
            )),
        )
    }

    fn request_provider_options(
        req: &ChatRequest,
    ) -> Option<&serde_json::Map<String, serde_json::Value>> {
        req.provider_options_map
            .get("groq")
            .and_then(|value| value.as_object())
    }

    fn provider_option_bool(req: &ChatRequest, camel_key: &str, snake_key: &str) -> Option<bool> {
        Self::request_provider_options(req)
            .and_then(|options| options.get(camel_key).or_else(|| options.get(snake_key)))
            .and_then(serde_json::Value::as_bool)
    }

    fn effective_structured_outputs(&self, req: &ChatRequest) -> bool {
        Self::provider_option_bool(req, "structuredOutputs", "structured_outputs")
            .or(self.default_structured_outputs)
            .unwrap_or(true)
    }

    fn structured_outputs_warning(&self, req: &ChatRequest) -> Option<Warning> {
        if self.effective_structured_outputs(req)
            || !matches!(
                req.response_format,
                Some(crate::types::chat::ResponseFormat::Json { .. })
            )
        {
            return None;
        }

        Some(Warning::unsupported(
            "responseFormat",
            Some(STRUCTURED_OUTPUTS_WARNING_DETAILS),
        ))
    }

    fn runtime_warnings(&self, req: &ChatRequest) -> Vec<Warning> {
        let mut warnings = Vec::new();

        if let Some(warning) = self.structured_outputs_warning(req) {
            warnings.push(warning);
        }

        if Self::has_browser_search_tool(req) {
            let model = req.common_params.model.trim();
            if !Self::supports_browser_search(model) {
                warnings.push(Self::unsupported_warning(model));
            }
        }

        warnings
    }

    fn maybe_insert_tool_choice(
        body_obj: &mut serde_json::Map<String, serde_json::Value>,
        choice: &ToolChoice,
    ) {
        body_obj
            .entry("tool_choice".to_string())
            .or_insert_with(|| crate::standards::openai::utils::convert_tool_choice(choice));
    }

    fn take_any(
        body_obj: &mut serde_json::Map<String, serde_json::Value>,
        keys: &[&str],
    ) -> Option<serde_json::Value> {
        for key in keys {
            if let Some(value) = body_obj.remove(*key) {
                return Some(value);
            }
        }
        None
    }

    fn normalize_groq_option_keys(
        &self,
        req: &ChatRequest,
        body_obj: &mut serde_json::Map<String, serde_json::Value>,
    ) {
        let structured_outputs =
            Self::take_any(body_obj, &["structuredOutputs", "structured_outputs"])
                .and_then(|value| value.as_bool());
        let strict_json_schema =
            Self::take_any(body_obj, &["strictJsonSchema", "strict_json_schema"])
                .and_then(|value| value.as_bool());

        if let Some(top_logprobs) = Self::take_any(body_obj, &["topLogprobs", "top_logprobs"]) {
            body_obj.insert("top_logprobs".to_string(), top_logprobs);
        }
        if let Some(service_tier) = Self::take_any(body_obj, &["serviceTier", "service_tier"]) {
            body_obj.insert("service_tier".to_string(), service_tier);
        }
        if let Some(reasoning_effort) =
            Self::take_any(body_obj, &["reasoningEffort", "reasoning_effort"])
        {
            body_obj.insert("reasoning_effort".to_string(), reasoning_effort);
        }
        if let Some(reasoning_format) =
            Self::take_any(body_obj, &["reasoningFormat", "reasoning_format"])
        {
            body_obj.insert("reasoning_format".to_string(), reasoning_format);
        }
        if let Some(parallel_tool_calls) =
            Self::take_any(body_obj, &["parallelToolCalls", "parallel_tool_calls"])
        {
            body_obj.insert("parallel_tool_calls".to_string(), parallel_tool_calls);
        }

        let effective_structured_outputs = structured_outputs
            .or(Self::provider_option_bool(
                req,
                "structuredOutputs",
                "structured_outputs",
            ))
            .or(self.default_structured_outputs)
            .unwrap_or(true);

        if !effective_structured_outputs
            && body_obj
                .get("response_format")
                .and_then(|value| value.get("type"))
                .and_then(|value| value.as_str())
                == Some("json_schema")
        {
            body_obj.insert(
                "response_format".to_string(),
                serde_json::json!({ "type": "json_object" }),
            );
        }

        if let Some(strict) = strict_json_schema
            && let Some(serde_json::Value::Object(response_format)) =
                body_obj.get_mut("response_format")
            && response_format.get("type").and_then(|value| value.as_str()) == Some("json_schema")
            && let Some(serde_json::Value::Object(json_schema)) =
                response_format.get_mut("json_schema")
        {
            json_schema.insert("strict".to_string(), serde_json::Value::Bool(strict));
        }
    }
}

impl LanguageModelMiddleware for GroqRequestMiddleware {
    fn transform_json_body(
        &self,
        req: &ChatRequest,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        let Some(body_obj) = body.as_object_mut() else {
            return Ok(());
        };

        self.normalize_groq_option_keys(req, body_obj);

        if !Self::has_browser_search_tool(req)
            || !Self::supports_browser_search(req.common_params.model.trim())
        {
            return Ok(());
        }

        let tools_value = body_obj
            .entry("tools".to_string())
            .or_insert_with(|| serde_json::Value::Array(Vec::new()));
        let Some(tools_array) = tools_value.as_array_mut() else {
            return Ok(());
        };

        let has_browser_search = tools_array.iter().any(|tool| {
            tool.get("type").and_then(|value| value.as_str()) == Some(BROWSER_SEARCH_TOOL_TYPE)
        });

        if !has_browser_search {
            tools_array.push(serde_json::json!({ "type": BROWSER_SEARCH_TOOL_TYPE }));
        }

        if let Some(choice) = req.tool_choice.as_ref() {
            Self::maybe_insert_tool_choice(body_obj, choice);
        }

        Ok(())
    }

    fn post_generate(
        &self,
        req: &ChatRequest,
        mut resp: ChatResponse,
    ) -> Result<ChatResponse, LlmError> {
        let warnings = self.runtime_warnings(req);
        if warnings.is_empty() {
            return Ok(resp);
        }

        match resp.warnings.as_mut() {
            Some(existing) => existing.extend(warnings),
            None => resp.warnings = Some(warnings),
        }

        Ok(resp)
    }

    fn on_stream_event(
        &self,
        req: &ChatRequest,
        ev: ChatStreamEvent,
    ) -> Result<Vec<ChatStreamEvent>, LlmError> {
        match ev {
            ChatStreamEvent::StreamEnd { mut response } => {
                let warnings = self.runtime_warnings(req);
                if !warnings.is_empty() {
                    match response.warnings.as_mut() {
                        Some(existing) => existing.extend(warnings),
                        None => response.warnings = Some(warnings),
                    }
                }

                Ok(vec![ChatStreamEvent::StreamEnd { response }])
            }
            other => Ok(vec![other]),
        }
    }
}
