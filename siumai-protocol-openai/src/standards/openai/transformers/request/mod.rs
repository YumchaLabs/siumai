//! Request transformers for OpenAI-compatible protocol (Chat/Embedding/Images) and OpenAI Responses API

use crate::error::LlmError;
use crate::execution::transformers::request::{
    Condition, GenericRequestTransformer, MappingProfile, ProviderParamsMergeStrategy,
    ProviderRequestHooks, RangeMode, Rule,
};
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
    ModerationRequest, RerankRequest,
};
use reqwest::multipart::{Form, Part};

#[derive(Clone)]
pub struct OpenAiRequestTransformer;

impl RequestTransformer for OpenAiRequestTransformer {
    fn provider_id(&self) -> &str {
        "openai"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        // Minimal provider-agnostic validation
        if req.common_params.model.is_empty() {
            return Err(LlmError::InvalidParameter(
                "Model must be specified".to_string(),
            ));
        }

        // Build via GenericRequestTransformer (profile + hooks)
        struct OpenAiChatHooks;
        impl ProviderRequestHooks for OpenAiChatHooks {
            fn build_base_chat_body(
                &self,
                req: &ChatRequest,
            ) -> Result<serde_json::Value, LlmError> {
                use crate::standards::openai::utils::convert_messages;
                let mut body = serde_json::json!({ "model": req.common_params.model });
                if let Some(t) = req.common_params.temperature {
                    body["temperature"] = serde_json::json!(t);
                }
                if let Some(tp) = req.common_params.top_p {
                    body["top_p"] = serde_json::json!(tp);
                }
                // Prefer max_completion_tokens for o1/o3 models, fallback to max_tokens
                if let Some(max) = req.common_params.max_completion_tokens {
                    body["max_completion_tokens"] = serde_json::json!(max);
                } else if let Some(max) = req.common_params.max_tokens {
                    body["max_tokens"] = serde_json::json!(max);
                }
                if let Some(seed) = req.common_params.seed {
                    body["seed"] = serde_json::json!(seed);
                }
                if let Some(stops) = &req.common_params.stop_sequences {
                    body["stop_sequences"] = serde_json::json!(stops);
                }

                let messages = convert_messages(&req.messages)?;
                body["messages"] = serde_json::to_value(messages)?;

                if let Some(tools) = &req.tools
                    && !tools.is_empty()
                {
                    let openai_tools =
                        crate::standards::openai::utils::convert_tools_to_openai_format(tools)?;
                    if !openai_tools.is_empty() {
                        body["tools"] = serde_json::Value::Array(openai_tools);

                        // Add tool_choice if specified
                        if let Some(choice) = &req.tool_choice {
                            body["tool_choice"] =
                                crate::standards::openai::utils::convert_tool_choice(choice);
                        }
                    }
                }

                if req.stream {
                    body["stream"] = serde_json::Value::Bool(true);
                    body["stream_options"] = serde_json::json!({ "include_usage": true });
                }
                Ok(body)
            }

            fn post_process_chat(
                &self,
                _req: &ChatRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // All provider-specific features are now handled via provider_options
                // in ProviderSpec::chat_before_send()
                Ok(())
            }
        }

        let profile = MappingProfile {
            provider_id: "openai",
            rules: vec![
                // stop_sequences -> stop (OpenAI specific)
                Rule::Move {
                    from: "stop_sequences",
                    to: "stop",
                },
                // For o1-* models, prefer max_completion_tokens by moving max_tokens
                Rule::When {
                    condition: Condition::ModelPrefix("o1-"),
                    rules: vec![
                        Rule::Move {
                            from: "max_tokens",
                            to: "max_completion_tokens",
                        },
                        Rule::Drop {
                            field: "max_tokens",
                        },
                    ],
                },
                // Stable ranges: temperature and top_p
                Rule::Range {
                    field: "temperature",
                    min: 0.0,
                    max: 2.0,
                    mode: RangeMode::Error,
                    message: Some("temperature must be between 0.0 and 2.0"),
                },
                Rule::Range {
                    field: "top_p",
                    min: 0.0,
                    max: 1.0,
                    mode: RangeMode::Error,
                    message: Some("top_p must be between 0.0 and 1.0"),
                },
                // Model condition: o1-* models forbid temperature and top_p
                Rule::ForbidWhen {
                    field: "temperature",
                    condition: Condition::ModelPrefix("o1-"),
                    message: "o1 models do not support temperature parameter",
                },
                Rule::ForbidWhen {
                    field: "top_p",
                    condition: Condition::ModelPrefix("o1-"),
                    message: "o1 models do not support top_p parameter",
                },
                // Tools upper bound (stable, per official docs)
                Rule::MaxLen {
                    field: "tools",
                    max: 128,
                    message: "OpenAI supports maximum 128 tools per request",
                },
            ],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };

        let generic = GenericRequestTransformer {
            profile,
            hooks: OpenAiChatHooks,
        };
        generic.transform_chat(req)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        // Reuse GenericRequestTransformer via embedding hooks (behavior unchanged)
        struct OpenAiEmbeddingHooks;
        impl ProviderRequestHooks for OpenAiEmbeddingHooks {
            fn build_base_embedding_body(
                &self,
                req: &EmbeddingRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let model = req
                    .model
                    .clone()
                    .unwrap_or_else(|| "text-embedding-3-small".to_string());
                let encoding_format = req.encoding_format.as_ref().map(|f| match f {
                    crate::types::EmbeddingFormat::Float => "float".to_string(),
                    crate::types::EmbeddingFormat::Base64 => "base64".to_string(),
                });
                let mut json = serde_json::json!({ "input": req.input, "model": model });
                if let Some(fmt) = encoding_format {
                    json["encoding_format"] = serde_json::json!(fmt);
                }
                if let Some(dim) = req.dimensions {
                    json["dimensions"] = serde_json::json!(dim);
                }
                if let Some(user) = &req.user {
                    json["user"] = serde_json::json!(user);
                }
                Ok(json)
            }
        }

        let hooks = OpenAiEmbeddingHooks;
        let profile = MappingProfile {
            provider_id: "openai",
            rules: vec![],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_embedding(req)
    }

    fn transform_image(
        &self,
        request: &ImageGenerationRequest,
    ) -> Result<serde_json::Value, LlmError> {
        // Use Generic transformer with image hooks (preserve existing behavior)
        struct OpenAiImageHooks;
        impl ProviderRequestHooks for OpenAiImageHooks {
            fn build_base_image_body(
                &self,
                req: &ImageGenerationRequest,
            ) -> Result<serde_json::Value, LlmError> {
                let mut body = serde_json::json!({ "prompt": req.prompt });
                if let Some(n) = Some(req.count).filter(|c| *c > 0) {
                    body["n"] = serde_json::json!(n);
                }
                if let Some(size) = &req.size {
                    body["size"] = serde_json::json!(size);
                }
                if let Some(q) = &req.quality {
                    body["quality"] = serde_json::json!(q);
                }
                if let Some(style) = &req.style {
                    body["style"] = serde_json::json!(style);
                }
                if let Some(fmt) = &req.response_format {
                    body["response_format"] = serde_json::json!(fmt);
                }
                if let Some(model) = &req.model {
                    body["model"] = serde_json::json!(model);
                }
                if let Some(neg) = &req.negative_prompt {
                    body["negative_prompt"] = serde_json::json!(neg);
                }
                Ok(body)
            }

            fn post_process_image(
                &self,
                _req: &ImageGenerationRequest,
                _body: &mut serde_json::Value,
            ) -> Result<(), LlmError> {
                // Merging extra parameters is handled by GenericRequestTransformer via merge_strategy
                Ok(())
            }
        }

        let hooks = OpenAiImageHooks;
        let profile = MappingProfile {
            provider_id: "openai",
            rules: vec![],
            merge_strategy: ProviderParamsMergeStrategy::Flatten,
        };
        let generic = GenericRequestTransformer { profile, hooks };
        generic.transform_image(request)
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        // Build multipart form for OpenAI Images Edit
        let mut form = Form::new().text("prompt", req.prompt.clone());
        let image_mime = crate::utils::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)?;
        form = form.part("image", image_part);
        if let Some(mask) = &req.mask {
            let mask_mime = crate::utils::guess_mime(Some(mask), None);
            let mask_part = Part::bytes(mask.clone())
                .file_name("mask")
                .mime_str(&mask_mime)?;
            form = form.part("mask", mask_part);
        }
        if let Some(size) = &req.size {
            form = form.text("size", size.clone());
        }
        if let Some(n) = req.count {
            form = form.text("n", n.to_string());
        }
        if let Some(fmt) = &req.response_format {
            form = form.text("response_format", fmt.clone());
        }
        Ok(ImageHttpBody::Multipart(form))
    }

    fn transform_image_variation(
        &self,
        req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        // Build multipart form for OpenAI Images Variation
        let mut form = Form::new();
        let image_mime = crate::utils::guess_mime(Some(&req.image), None);
        let image_part = Part::bytes(req.image.clone())
            .file_name("image")
            .mime_str(&image_mime)?;
        form = form.part("image", image_part);
        if let Some(size) = &req.size {
            form = form.text("size", size.clone());
        }
        if let Some(n) = req.count {
            form = form.text("n", n.to_string());
        }
        if let Some(fmt) = &req.response_format {
            form = form.text("response_format", fmt.clone());
        }
        Ok(ImageHttpBody::Multipart(form))
    }

    fn transform_rerank(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        let mut payload = serde_json::json!({
            "model": req.model,
            "query": req.query,
            "documents": req.documents,
        });
        if let Some(instr) = &req.instruction {
            payload["instruction"] = serde_json::json!(instr);
        }
        if let Some(top_n) = req.top_n {
            payload["top_n"] = serde_json::json!(top_n);
        }
        if let Some(rd) = req.return_documents {
            payload["return_documents"] = serde_json::json!(rd);
        }
        if let Some(maxc) = req.max_chunks_per_doc {
            payload["max_chunks_per_doc"] = serde_json::json!(maxc);
        }
        if let Some(over) = req.overlap_tokens {
            payload["overlap_tokens"] = serde_json::json!(over);
        }
        Ok(payload)
    }

    fn transform_moderation(&self, req: &ModerationRequest) -> Result<serde_json::Value, LlmError> {
        let model = req
            .model
            .clone()
            .unwrap_or_else(|| "text-moderation-latest".to_string());

        let input_value = if let Some(arr) = &req.inputs {
            serde_json::Value::Array(
                arr.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            )
        } else {
            serde_json::Value::String(req.input.clone())
        };

        Ok(serde_json::json!({ "model": model, "input": input_value }))
    }
}

#[cfg(test)]
mod tests_openai_rules {
    use super::*;

    #[test]
    fn when_model_prefix_o1_moves_max_tokens() {
        let tx = OpenAiRequestTransformer;
        let mut req = ChatRequest::new(vec![]);
        req.common_params.model = "o1-mini".to_string();
        req.common_params.max_tokens = Some(123);
        let body = tx.transform_chat(&req).expect("transform");
        // max_tokens should be moved to max_completion_tokens
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], serde_json::json!(123));
    }
}

// Tests for structured_output via provider_params have been removed
// as this functionality is now handled via provider_options in ProviderSpec::chat_before_send()

#[cfg(feature = "openai-responses")]
mod responses;

#[cfg(feature = "openai-responses")]
pub use responses::OpenAiResponsesRequestTransformer;
