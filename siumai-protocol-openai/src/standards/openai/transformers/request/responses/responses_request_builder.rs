use crate::error::LlmError;
use crate::execution::transformers::request::ProviderRequestHooks;
use crate::types::ChatRequest;

use super::{
    OpenAiResponsesRequestTransformer, ResponsesInputConversionState, provider_option_object,
    rename_json_field,
};

/// OpenAI Responses request-body builder.
///
/// This module owns the top-level Responses API body construction and typed
/// provider-option post-processing. Item-level prompt conversion remains in the
/// parent transformer until a later seam is split.
pub(super) struct ResponsesRequestHooks;

impl ProviderRequestHooks for ResponsesRequestHooks {
    fn build_base_chat_body(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let mut body = serde_json::json!({
            "model": req.common_params.model,
        });

        if req.stream {
            body["stream"] = serde_json::Value::Bool(true);
        }

        let mut input_items: Vec<serde_json::Value> = Vec::new();
        let mut state = ResponsesInputConversionState::default();
        for message in &req.messages {
            OpenAiResponsesRequestTransformer::extend_message(
                req,
                message,
                &mut state,
                &mut input_items,
            )?;
        }
        body["input"] = serde_json::Value::Array(input_items);

        if let Some(tools) = &req.tools {
            let openai_tools =
                crate::standards::openai::utils::convert_tools_to_responses_format(tools)?;
            if !openai_tools.is_empty() {
                body["tools"] = serde_json::Value::Array(openai_tools);

                if let Some(choice) = &req.tool_choice
                    && let Some(tool_choice) =
                        crate::standards::openai::utils::convert_responses_tool_choice(
                            choice,
                            req.tools.as_deref(),
                        )
                {
                    body["tool_choice"] = tool_choice;
                }
            }
        }

        if let Some(temp) = req.common_params.temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        if let Some(top_p) = req.common_params.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }

        if let Some(max_tokens) = req.common_params.max_completion_tokens {
            body["max_output_tokens"] = serde_json::json!(max_tokens);
        } else if let Some(max_tokens) = req.common_params.max_tokens {
            body["max_output_tokens"] = serde_json::json!(max_tokens);
        }

        if let Some(format) = &req.response_format {
            let text = body
                .as_object_mut()
                .expect("responses request body must be an object")
                .entry("text".to_string())
                .or_insert_with(|| serde_json::json!({}));

            if !text.is_object() {
                *text = serde_json::json!({});
            }

            text.as_object_mut()
                .expect("responses text entry was normalized to an object")
                .insert(
                    "format".to_string(),
                    crate::standards::openai::utils::convert_responses_response_format(format),
                );
        }

        Ok(body)
    }

    fn post_process_chat(
        &self,
        req: &crate::types::ChatRequest,
        body: &mut serde_json::Value,
    ) -> Result<(), LlmError> {
        let xai_options = req.provider_options_map.get_object("xai");
        let responses_options = xai_options.or_else(|| {
            provider_option_object(Some(&req.provider_options_map), "openai")
                .or_else(|| provider_option_object(Some(&req.provider_options_map), "azure"))
        });

        let Some(responses_options) = responses_options else {
            return Ok(());
        };

        let Some(body_obj) = body.as_object_mut() else {
            return Ok(());
        };

        let get_option = |camel_case: &str, snake_case: &str| {
            responses_options
                .get(camel_case)
                .or_else(|| responses_options.get(snake_case))
        };
        let get_responses_api_option = |camel_case: &str, snake_case: &str| {
            get_option("responsesApi", "responses_api")
                .and_then(|value| value.as_object())
                .and_then(|options| options.get(camel_case).or_else(|| options.get(snake_case)))
        };

        if xai_options.is_some() {
            let reasoning_effort =
                get_option("reasoningEffort", "reasoning_effort").and_then(|value| value.as_str());
            let reasoning_summary = get_option("reasoningSummary", "reasoning_summary")
                .and_then(|value| value.as_str());

            if reasoning_effort.is_some() || reasoning_summary.is_some() {
                let reasoning = body_obj
                    .entry("reasoning".to_string())
                    .or_insert_with(|| serde_json::json!({}));
                if !reasoning.is_object() {
                    *reasoning = serde_json::json!({});
                }
                let reasoning_obj = reasoning
                    .as_object_mut()
                    .expect("xai reasoning body was normalized to an object");
                if let Some(effort) = reasoning_effort {
                    reasoning_obj.insert("effort".to_string(), serde_json::json!(effort));
                }
                if let Some(summary) = reasoning_summary {
                    reasoning_obj.insert("summary".to_string(), serde_json::json!(summary));
                }
            }

            let top_logprobs = get_option("topLogprobs", "top_logprobs").cloned();
            let logprobs = get_option("logprobs", "logprobs").and_then(|value| value.as_bool());
            if let Some(top_logprobs) = top_logprobs {
                body_obj.insert("top_logprobs".to_string(), top_logprobs);
                body_obj.insert("logprobs".to_string(), serde_json::json!(true));
            } else if let Some(logprobs) = logprobs {
                body_obj.insert("logprobs".to_string(), serde_json::json!(logprobs));
            }
        }

        let store = get_option("store", "store").and_then(|value| value.as_bool());
        if store == Some(false) {
            body_obj.insert("store".to_string(), serde_json::json!(false));
        }

        if let Some(previous_response_id) = get_option("previousResponseId", "previous_response_id")
            .and_then(|value| value.as_str())
        {
            body_obj.insert(
                "previous_response_id".to_string(),
                serde_json::json!(previous_response_id),
            );
        }

        let include_value = get_option("include", "include");
        let include_was_explicit_array = include_value.is_some_and(|value| value.is_array());
        let mut include = include_value
            .and_then(|value| value.as_array())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str().map(|value| value.to_string()))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if store == Some(false)
            && !include
                .iter()
                .any(|value| value == "reasoning.encrypted_content")
        {
            include.push("reasoning.encrypted_content".to_string());
        }

        if include_was_explicit_array || !include.is_empty() {
            body_obj.insert("include".to_string(), serde_json::json!(include));
        }

        let context_management_value = get_option("contextManagement", "context_management");
        let context_management_was_explicit_array =
            context_management_value.is_some_and(|value| value.is_array());
        let context_management = context_management_value
            .and_then(|value| value.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| {
                        let mut obj = item.as_object()?.clone();
                        rename_json_field(&mut obj, "compactThreshold", "compact_threshold");
                        Some(serde_json::Value::Object(obj))
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        if context_management_was_explicit_array {
            body_obj.insert(
                "context_management".to_string(),
                serde_json::Value::Array(context_management),
            );
        }

        let has_wire_tools = body_obj
            .get("tools")
            .and_then(|value| value.as_array())
            .is_some_and(|tools| !tools.is_empty());
        if xai_options.is_none() && has_wire_tools {
            let allowed_tools = get_option("allowedTools", "allowed_tools")
                .or_else(|| get_responses_api_option("allowedTools", "allowed_tools"));
            if let Some(tool_choice) =
                OpenAiResponsesRequestTransformer::allowed_tools_choice(req, allowed_tools)
            {
                body_obj.insert("tool_choice".to_string(), tool_choice);
            }
        }

        Ok(())
    }
}
