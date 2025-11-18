//! OpenAI Responses API bridge helpers
//!
//! This module contains aggregator-level helpers that convert
//! `ChatRequest` + typed `ProviderOptions::OpenAi` into the core
//! `ResponsesInput` type used by the std-openai crate. It keeps
//! the detailed JSON shape construction close to the OpenAI
//! provider while allowing `OpenAiSpec` to stay focused on
//! high-level wiring.

use crate::error::LlmError;
use crate::types::{ChatRequest, ProviderOptions};
use siumai_core::execution::responses::ResponsesInput;
use std::collections::HashMap;

/// Build a core `ResponsesInput` from an aggregator-level `ChatRequest`.
///
/// This helper encapsulates all mapping logic that was previously
/// inlined inside `OpenAiSpec::choose_chat_transformers` for the
/// Responses API path. It keeps behavior identical while making
/// the spec implementation easier to read and maintain.
pub fn build_responses_input(req: &ChatRequest) -> Result<ResponsesInput, LlmError> {
    // Reuse existing message → input-item mapping logic.
    let mut input_items = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        let item =
            crate::providers::openai::transformers::OpenAiResponsesRequestTransformer::convert_message(
                m,
            )?;
        input_items.push(item);
    }

    let mut extra: HashMap<String, serde_json::Value> = HashMap::new();

    // Core Responses fields that used to live in the base body.
    extra.insert("stream".to_string(), serde_json::Value::Bool(req.stream));

    // tools/tool_choice in Responses format
    if let Some(tools) = &req.tools {
        let openai_tools =
            crate::providers::openai::utils::convert_tools_to_responses_format(tools)?;
        if !openai_tools.is_empty() {
            extra.insert("tools".to_string(), serde_json::Value::Array(openai_tools));

            if let Some(choice) = &req.tool_choice {
                let tc = crate::providers::openai::utils::convert_tool_choice(choice);
                extra.insert("tool_choice".to_string(), tc);
            }
        }
    }

    // stream_options
    if req.stream {
        extra.insert(
            "stream_options".to_string(),
            serde_json::json!({ "include_usage": true }),
        );
    }

    // temperature range-checked by GenericRequestTransformer previously;
    // we keep the value here and let upstream validation handle ranges.
    if let Some(temp) = req.common_params.temperature {
        extra.insert("temperature".to_string(), serde_json::json!(temp));
    }

    // max_output_tokens (prefer max_completion_tokens, fallback to max_tokens)
    if let Some(max_tokens) = req.common_params.max_completion_tokens {
        extra.insert(
            "max_output_tokens".to_string(),
            serde_json::json!(max_tokens),
        );
    } else if let Some(max_tokens) = req.common_params.max_tokens {
        extra.insert(
            "max_output_tokens".to_string(),
            serde_json::json!(max_tokens),
        );
    }

    // seed
    if let Some(seed) = req.common_params.seed {
        extra.insert("seed".to_string(), serde_json::json!(seed));
    }

    // Inject OpenAI-specific ProviderOptions into ResponsesInput::extra
    if let ProviderOptions::OpenAi(ref options) = req.provider_options {
        // Responses API configuration
        if let Some(ref cfg) = options.responses_api {
            if let Some(ref pid) = cfg.previous_response_id {
                extra.insert("previous_response_id".to_string(), serde_json::json!(pid));
            }
            if let Some(ref fmt) = cfg.response_format {
                extra.insert("response_format".to_string(), fmt.clone());
            }
            if let Some(bg) = cfg.background {
                extra.insert("background".to_string(), serde_json::json!(bg));
            }
            if let Some(ref inc) = cfg.include {
                extra.insert("include".to_string(), serde_json::json!(inc));
            }
            if let Some(ref instr) = cfg.instructions {
                extra.insert("instructions".to_string(), serde_json::json!(instr));
            }
            if let Some(mtc) = cfg.max_tool_calls {
                extra.insert("max_tool_calls".to_string(), serde_json::json!(mtc));
            }
            if let Some(st) = cfg.store {
                extra.insert("store".to_string(), serde_json::json!(st));
            }
            if let Some(ref trunc) = cfg.truncation
                && let Ok(val) = serde_json::to_value(trunc)
            {
                extra.insert("truncation".to_string(), val);
            }
            if let Some(ref verb) = cfg.text_verbosity
                && let Ok(val) = serde_json::to_value(verb)
            {
                // text_verbosity is nested under "text.verbosity"
                let existing_text = extra
                    .remove("text")
                    .unwrap_or_else(|| serde_json::json!({}));
                let mut map = existing_text.as_object().cloned().unwrap_or_default();
                map.insert("verbosity".to_string(), val);
                extra.insert("text".to_string(), serde_json::Value::Object(map));
            }
            if let Some(ref meta) = cfg.metadata {
                extra.insert("metadata".to_string(), serde_json::json!(meta));
            }
            if let Some(ptc) = cfg.parallel_tool_calls {
                extra.insert("parallel_tool_calls".to_string(), serde_json::json!(ptc));
            }
        }

        // Reasoning effort / service tier
        if let Some(effort) = options.reasoning_effort
            && let Ok(val) = serde_json::to_value(effort)
        {
            extra.insert("reasoning_effort".to_string(), val);
        }
        if let Some(tier) = options.service_tier
            && let Ok(val) = serde_json::to_value(tier)
        {
            extra.insert("service_tier".to_string(), val);
        }

        // Modalities (for multimodal/audio output)
        if let Some(ref mods) = options.modalities
            && let Ok(val) = serde_json::to_value(mods)
        {
            extra.insert("modalities".to_string(), val);
        }

        // Audio configuration
        if let Some(ref aud) = options.audio
            && let Ok(val) = serde_json::to_value(aud)
        {
            extra.insert("audio".to_string(), val);
        }

        // Prediction (Predicted Outputs)
        if let Some(ref pred) = options.prediction
            && let Ok(val) = serde_json::to_value(pred)
        {
            extra.insert("prediction".to_string(), val);
        }

        // Web search options
        if let Some(ref ws) = options.web_search_options
            && let Ok(val) = serde_json::to_value(ws)
        {
            extra.insert("web_search_options".to_string(), val);
        }
    }

    Ok(ResponsesInput {
        model: req.common_params.model.clone(),
        input: input_items,
        extra,
    })
}
