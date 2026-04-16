use super::*;

#[cfg(test)]
mod image_block_tests {
    use super::*;

    #[test]
    fn tool_result_image_content_includes_media_type() {
        let png_bytes: Vec<u8> = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR".to_vec();

        let tool_result = ContentPart::ToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: "generate_image".to_string(),
            output: ToolResultOutput::content(vec![ToolResultContentPart::image_binary(
                png_bytes,
                "image/png",
            )]),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: None,
        };

        let content = MessageContent::MultiModal(vec![tool_result]);
        let mapped = convert_message_content(&content).expect("convert ok");

        let arr = mapped.as_array().expect("array");
        let tool_result_obj = arr[0].as_object().expect("object");
        assert_eq!(
            tool_result_obj.get("type").and_then(|v| v.as_str()),
            Some("tool_result")
        );

        let inner = tool_result_obj
            .get("content")
            .and_then(|v| v.as_array())
            .expect("tool_result content array");
        let image = inner[0].as_object().expect("image block");
        assert_eq!(image.get("type").and_then(|v| v.as_str()), Some("image"));

        let media_type = image
            .get("source")
            .and_then(|v| v.get("media_type"))
            .and_then(|v| v.as_str())
            .expect("media_type");
        assert_eq!(media_type, "image/png");
    }

    #[test]
    fn tool_result_json_is_stringified() {
        let tool_result = ContentPart::ToolResult {
            tool_call_id: "call_1".to_string(),
            tool_name: "test".to_string(),
            output: ToolResultOutput::Json {
                value: serde_json::json!({ "ok": true }),
                provider_options: crate::types::ProviderOptionsMap::default(),
            },
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: None,
        };

        let content = MessageContent::MultiModal(vec![tool_result]);
        let mapped = convert_message_content(&content).expect("convert ok");

        let arr = mapped.as_array().expect("array");
        let tool_result_obj = arr[0].as_object().expect("object");
        let content_value = tool_result_obj.get("content").expect("content");
        assert!(
            content_value.is_string(),
            "expected stringified JSON, got: {content_value:?}"
        );
    }
}

#[cfg(test)]
mod document_provider_options_tests {
    use super::*;

    #[test]
    fn file_part_provider_options_control_document_citations_and_title() {
        let mut provider_options = crate::types::ProviderOptionsMap::default();
        provider_options.insert(
            "anthropic",
            serde_json::json!({
                "citations": { "enabled": true },
                "title": "My Doc",
                "context": "background",
            }),
        );

        let part = ContentPart::File {
            source: FilePartSource::binary(b"%PDF-1.7".to_vec()),
            media_type: "application/pdf".to_string(),
            filename: Some("fallback.pdf".to_string()),
            provider_options,
            provider_metadata: None,
        };

        let content = MessageContent::MultiModal(vec![part]);
        let mapped = convert_message_content(&content).expect("convert ok");
        let arr = mapped.as_array().expect("array");

        assert_eq!(arr[0]["type"], "document");
        assert_eq!(arr[0]["title"], "My Doc");
        assert_eq!(arr[0]["context"], "background");
        assert_eq!(arr[0]["citations"]["enabled"], true);
    }

    #[test]
    fn file_part_provider_metadata_is_ignored_for_request_document_settings() {
        let mut provider_metadata = std::collections::HashMap::new();
        provider_metadata.insert(
            "anthropic".to_string(),
            serde_json::json!({
                "citations": { "enabled": true },
                "title": "Legacy Doc",
                "context": "legacy background",
            }),
        );

        let part = ContentPart::File {
            source: FilePartSource::binary(b"%PDF-1.7".to_vec()),
            media_type: "application/pdf".to_string(),
            filename: Some("fallback.pdf".to_string()),
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: Some(provider_metadata),
        };

        let content = MessageContent::MultiModal(vec![part]);
        let mapped = convert_message_content(&content).expect("convert ok");
        let arr = mapped.as_array().expect("array");

        assert_eq!(arr[0]["type"], "document");
        assert_eq!(arr[0]["title"], "fallback.pdf");
        assert!(arr[0].get("context").is_none());
        assert!(arr[0].get("citations").is_none());
    }

    #[test]
    fn tool_result_custom_content_maps_to_tool_reference() {
        let tool_result = ContentPart::ToolResult {
            tool_call_id: "search-1".to_string(),
            tool_name: "search_tools".to_string(),
            output: ToolResultOutput::content(vec![
                ToolResultContentPart::custom().with_provider_option(
                    "anthropic",
                    serde_json::json!({
                        "type": "tool-reference",
                        "toolName": "get_weather",
                    }),
                ),
            ]),
            input: None,
            provider_executed: None,
            dynamic: None,
            preliminary: None,
            title: None,
            provider_options: crate::types::ProviderOptionsMap::default(),
            provider_metadata: None,
        };

        let content = MessageContent::MultiModal(vec![tool_result]);
        let mapped = convert_message_content(&content).expect("convert ok");
        let arr = mapped.as_array().expect("array");
        assert_eq!(arr[0]["type"], "tool_result");
        assert_eq!(arr[0]["content"][0]["type"], "tool_reference");
        assert_eq!(arr[0]["content"][0]["tool_name"], "get_weather");
    }

    #[test]
    fn image_provider_reference_maps_to_anthropic_file_source() {
        let part = ContentPart::image_provider_reference(ProviderReference::single(
            "anthropic",
            "file-anthropic",
        ));

        let content = MessageContent::MultiModal(vec![part]);
        let mapped = convert_message_content(&content).expect("convert ok");
        let arr = mapped.as_array().expect("array");

        assert_eq!(arr[0]["type"], serde_json::json!("image"));
        assert_eq!(arr[0]["source"]["type"], serde_json::json!("file"));
        assert_eq!(
            arr[0]["source"]["file_id"],
            serde_json::json!("file-anthropic")
        );
    }

    #[test]
    fn image_provider_reference_requires_anthropic_entry() {
        let part =
            ContentPart::image_provider_reference(ProviderReference::single("openai", "file-1"));

        let content = MessageContent::MultiModal(vec![part]);
        let err = convert_message_content(&content).expect_err("expected missing provider");

        assert!(matches!(
            err,
            LlmError::InvalidParameter(message)
                if message
                    == "No provider reference found for provider 'anthropic'. Available providers: openai"
        ));
    }
}

pub fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
    #[allow(unreachable_patterns)]
    match content {
        MessageContent::Text(text) => Ok(serde_json::Value::Array(vec![serde_json::json!({
            "type": "text",
            "text": text
        })])),
        MessageContent::MultiModal(parts) => {
            let mut content_parts = Vec::new();

            for part in parts {
                match part {
                    ContentPart::Text { text, .. } => {
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    ContentPart::Image { source, .. } => {
                        // Anthropic requires base64-encoded images
                        let (media_type, data) = match source {
                            FilePartSource::Media(MediaSource::Base64 { data }) => {
                                let bytes = base64::engine::general_purpose::STANDARD
                                    .decode(data.as_bytes())
                                    .ok();
                                let media_type = bytes
                                    .as_deref()
                                    .map(super::headers::guess_image_media_type_from_bytes)
                                    .unwrap_or_else(|| "image/jpeg".to_string());
                                (media_type, data.clone())
                            }
                            FilePartSource::Media(MediaSource::Binary { data }) => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                let media_type =
                                    super::headers::guess_image_media_type_from_bytes(data);
                                (media_type, encoded)
                            }
                            FilePartSource::Media(MediaSource::Url { url }) => {
                                // Anthropic doesn't support URLs, convert to text
                                content_parts.push(serde_json::json!({
                                    "type": "text",
                                    "text": format!("[Image: {}]", url)
                                }));
                                continue;
                            }
                            FilePartSource::ProviderReference { provider_reference } => {
                                content_parts.push(serde_json::json!({
                                    "type": "image",
                                    "source": {
                                        "type": "file",
                                        "file_id": resolve_anthropic_provider_reference(provider_reference)?,
                                    }
                                }));
                                continue;
                            }
                        };

                        content_parts.push(serde_json::json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data
                            }
                        }));
                    }
                    ContentPart::Audio { source, .. } => {
                        // Anthropic does not support audio input
                        let placeholder = match source {
                            crate::types::chat::MediaSource::Url { url } => {
                                format!("[Audio: {}]", url)
                            }
                            _ => "[Audio input not supported by Anthropic]".to_string(),
                        };
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": placeholder
                        }));
                    }
                    ContentPart::Source { source, .. } => {
                        // Anthropic does not support `source` parts in request input.
                        // Convert them into a best-effort text placeholder to preserve context.
                        let text = match source {
                            crate::types::SourcePart::Url { url, title } => {
                                let title = title.as_deref().filter(|s| !s.is_empty());
                                if let Some(title) = title
                                    && title != url
                                {
                                    format!("[Source: {title} ({url})]")
                                } else {
                                    format!("[Source: {url}]")
                                }
                            }
                            crate::types::SourcePart::Document {
                                media_type,
                                title,
                                filename,
                            } => {
                                let title = title.as_str();
                                let filename = filename.as_deref().filter(|s| !s.is_empty());
                                let media_type =
                                    Some(media_type.as_str()).filter(|s| !s.is_empty());
                                match (filename, media_type) {
                                    (Some(filename), _) => {
                                        format!("[Source document: {title} ({filename})]")
                                    }
                                    (None, Some(media_type)) => {
                                        format!("[Source document: {title} [{media_type}]]")
                                    }
                                    _ => format!("[Source document: {title}]"),
                                }
                            }
                        };
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": text
                        }));
                    }
                    ContentPart::File {
                        source,
                        media_type,
                        filename,
                        provider_options,
                        ..
                    } => {
                        fn apply_anthropic_document_options(
                            doc: &mut serde_json::Value,
                            filename: &Option<String>,
                            provider_options: &crate::types::ProviderOptionsMap,
                        ) {
                            let anthropic_options = provider_options.get_object("anthropic");

                            let Some(obj) = anthropic_options else {
                                if let Some(name) = filename.as_ref() {
                                    doc["title"] = serde_json::json!(name);
                                }
                                return;
                            };

                            if let Some(title) = obj.get("title").and_then(|v| v.as_str()) {
                                if !title.is_empty() {
                                    doc["title"] = serde_json::json!(title);
                                }
                            } else if let Some(name) = filename.as_ref() {
                                doc["title"] = serde_json::json!(name);
                            }

                            if let Some(context) = obj.get("context").and_then(|v| v.as_str())
                                && !context.is_empty()
                            {
                                doc["context"] = serde_json::json!(context);
                            }

                            let enabled = obj
                                .get("citations")
                                .and_then(|v| v.as_object())
                                .and_then(|c| c.get("enabled"))
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            if enabled
                                && doc.get("type").and_then(|v| v.as_str()) == Some("document")
                            {
                                doc["citations"] = serde_json::json!({ "enabled": true });
                            }
                        }

                        // Anthropic supports document parts for PDFs and plain text.
                        if media_type == "application/pdf" {
                            let source_json = match source {
                                FilePartSource::Media(MediaSource::Base64 { data }) => {
                                    serde_json::json!({
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": data
                                    })
                                }
                                FilePartSource::Media(MediaSource::Binary { data }) => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    serde_json::json!({
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": encoded
                                    })
                                }
                                FilePartSource::Media(MediaSource::Url { url }) => {
                                    serde_json::json!({
                                        "type": "url",
                                        "url": url
                                    })
                                }
                                FilePartSource::ProviderReference { provider_reference } => {
                                    serde_json::json!({
                                        "type": "file",
                                        "file_id": resolve_anthropic_provider_reference(provider_reference)?,
                                    })
                                }
                            };

                            let mut doc = serde_json::json!({
                                "type": "document",
                                "source": source_json,
                            });
                            apply_anthropic_document_options(&mut doc, filename, provider_options);
                            content_parts.push(doc);
                        } else if media_type == "text/plain" {
                            let source_json = match source {
                                FilePartSource::Media(MediaSource::Url { url }) => {
                                    serde_json::json!({
                                        "type": "url",
                                        "url": url
                                    })
                                }
                                FilePartSource::Media(MediaSource::Binary { data }) => {
                                    let text =
                                        String::from_utf8(data.clone()).unwrap_or_else(|_| {
                                            "[Invalid UTF-8 text/plain]".to_string()
                                        });
                                    serde_json::json!({
                                        "type": "text",
                                        "media_type": "text/plain",
                                        "data": text
                                    })
                                }
                                FilePartSource::Media(MediaSource::Base64 { data }) => {
                                    let decoded = base64::engine::general_purpose::STANDARD
                                        .decode(data.as_bytes())
                                        .ok()
                                        .and_then(|bytes| String::from_utf8(bytes).ok())
                                        .unwrap_or_else(|| {
                                            "[Invalid base64 text/plain]".to_string()
                                        });
                                    serde_json::json!({
                                        "type": "text",
                                        "media_type": "text/plain",
                                        "data": decoded
                                    })
                                }
                                FilePartSource::ProviderReference { provider_reference } => {
                                    serde_json::json!({
                                        "type": "file",
                                        "file_id": resolve_anthropic_provider_reference(provider_reference)?,
                                    })
                                }
                            };

                            let mut doc = serde_json::json!({
                                "type": "document",
                                "source": source_json,
                            });
                            apply_anthropic_document_options(&mut doc, filename, provider_options);
                            content_parts.push(doc);
                        } else {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Unsupported file type: {}]", media_type)
                            }));
                        }
                    }
                    ContentPart::ReasoningFile { media_type, .. } => {
                        // Anthropic does not have a stable request-side equivalent for
                        // reasoning files in assistant prompts. Preserve a textual hint.
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": format!("[Reasoning file: {}]", media_type)
                        }));
                    }
                    ContentPart::Custom { kind, .. } => {
                        // Anthropic request conversion does not support arbitrary custom parts.
                        // Preserve a textual hint instead of dropping content silently.
                        content_parts.push(serde_json::json!({
                            "type": "text",
                            "text": format!("[Custom content: {}]", kind)
                        }));
                    }
                    ContentPart::ToolCall {
                        tool_call_id,
                        tool_name,
                        arguments,
                        ..
                    } => {
                        // Tool calls in Anthropic format
                        content_parts.push(serde_json::json!({
                            "type": "tool_use",
                            "id": tool_call_id,
                            "name": tool_name,
                            "input": arguments
                        }));
                    }
                    ContentPart::ToolResult {
                        tool_call_id,
                        output,
                        ..
                    } => {
                        // Tool results in Anthropic format
                        use crate::types::ToolResultOutput;

                        let (content_value, is_error) = match output {
                            ToolResultOutput::Text { value, .. } => {
                                (serde_json::json!(value), false)
                            }
                            // Anthropic Messages API requires tool_result `content` to be a string
                            // or an array of content blocks; JSON objects must be stringified.
                            ToolResultOutput::Json { value, .. } => (
                                serde_json::json!(serde_json::to_string(value).unwrap_or_default()),
                                false,
                            ),
                            ToolResultOutput::ErrorText { value, .. } => {
                                (serde_json::json!(value), true)
                            }
                            ToolResultOutput::ErrorJson { value, .. } => (
                                serde_json::json!(serde_json::to_string(value).unwrap_or_default()),
                                true,
                            ),
                            ToolResultOutput::ExecutionDenied { reason, .. } => {
                                let msg = reason
                                    .as_ref()
                                    .map(|r| format!("Execution denied: {}", r))
                                    .unwrap_or_else(|| "Execution denied".to_string());
                                (serde_json::json!(msg), true)
                            }
                            ToolResultOutput::Content { value, .. } => {
                                // Convert multimodal content to Anthropic format
                                let content_array: Vec<serde_json::Value> = value.iter().map(|part| {
                                    use crate::types::ToolResultContentPart;
                                    match part {
                                        ToolResultContentPart::Text { text, .. } => {
                                            serde_json::json!({"type": "text", "text": text})
                                        }
                                        ToolResultContentPart::ImageData {
                                            data,
                                            media_type,
                                            ..
                                        } => {
                                            serde_json::json!({
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": media_type,
                                                    "data": data
                                                }
                                            })
                                        }
                                        ToolResultContentPart::ImageUrl { url, .. } => {
                                            serde_json::json!({
                                                "type": "text",
                                                "text": format!("[Image: {}]", url)
                                            })
                                        }
                                        ToolResultContentPart::ImageFileId { .. } => {
                                            serde_json::json!({
                                                "type": "text",
                                                "text": "[Image file id attachment]"
                                            })
                                        }
                                        ToolResultContentPart::FileData {
                                            data,
                                            media_type,
                                            ..
                                        } if media_type == "application/pdf" => {
                                            serde_json::json!({
                                                "type": "document",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": media_type,
                                                    "data": data
                                                }
                                            })
                                        }
                                        ToolResultContentPart::FileData { media_type, .. } => {
                                            serde_json::json!({
                                                "type": "text",
                                                "text": format!("[Unsupported file attachment: {}]", media_type)
                                            })
                                        }
                                        ToolResultContentPart::FileUrl { url, .. } => {
                                            serde_json::json!({
                                                "type": "document",
                                                "source": {
                                                    "type": "url",
                                                    "url": url
                                                }
                                            })
                                        }
                                        ToolResultContentPart::FileId { .. } => {
                                            serde_json::json!({
                                                "type": "text",
                                                "text": "[File id attachment]"
                                            })
                                        }
                                        ToolResultContentPart::Custom { provider_options } => {
                                            let anthropic_options = provider_options
                                                .get_object("anthropic")
                                                .or_else(|| provider_options.get_object("claude"));

                                            if let Some(anthropic_options) = anthropic_options
                                                && anthropic_options
                                                    .get("type")
                                                    .and_then(|v| v.as_str())
                                                    == Some("tool-reference")
                                                && let Some(tool_name) = anthropic_options
                                                    .get("toolName")
                                                    .or_else(|| anthropic_options.get("tool_name"))
                                                    .and_then(|v| v.as_str())
                                            {
                                                serde_json::json!({
                                                    "type": "tool_reference",
                                                    "tool_name": tool_name,
                                                })
                                            } else {
                                                serde_json::json!({
                                                    "type": "text",
                                                    "text": "[Custom tool content]"
                                                })
                                            }
                                        }
                                    }
                                }).collect();
                                (serde_json::Value::Array(content_array), false)
                            }
                        };

                        content_parts.push(serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content_value,
                            "is_error": is_error
                        }));
                    }
                    ContentPart::ToolApprovalResponse { .. } => {}
                    ContentPart::ToolApprovalRequest { .. } => {}
                    ContentPart::Reasoning { text, .. } => {
                        // Emit as a thinking block (Anthropic format). `convert_messages` will
                        // keep it only when Anthropic replay metadata is present on the reasoning
                        // part; otherwise the block is omitted from assistant replay.
                        content_parts.push(serde_json::json!({
                            "type": "thinking",
                            "thinking": text
                        }));
                    }
                }
            }

            Ok(serde_json::Value::Array(content_parts))
        }
        _ => Ok(serde_json::Value::Array(vec![serde_json::json!({
            "type": "text",
            "text": content.all_text(),
        })])),
    }
}
