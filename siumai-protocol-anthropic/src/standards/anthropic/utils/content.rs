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
            output: ToolResultOutput::content(vec![ToolResultContentPart::Image {
                source: MediaSource::Binary { data: png_bytes },
                detail: None,
            }]),
            provider_executed: None,
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
            },
            provider_executed: None,
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
mod document_provider_metadata_tests {
    use super::*;

    #[test]
    fn file_part_provider_metadata_controls_document_citations_and_title() {
        let mut provider_metadata = std::collections::HashMap::new();
        provider_metadata.insert(
            "anthropic".to_string(),
            serde_json::json!({
                "citations": { "enabled": true },
                "title": "My Doc",
                "context": "background",
            }),
        );

        let part = ContentPart::File {
            source: MediaSource::Binary {
                data: b"%PDF-1.7".to_vec(),
            },
            media_type: "application/pdf".to_string(),
            filename: Some("fallback.pdf".to_string()),
            provider_metadata: Some(provider_metadata),
        };

        let content = MessageContent::MultiModal(vec![part]);
        let mapped = convert_message_content(&content).expect("convert ok");
        let arr = mapped.as_array().expect("array");

        assert_eq!(arr[0]["type"], "document");
        assert_eq!(arr[0]["title"], "My Doc");
        assert_eq!(arr[0]["context"], "background");
        assert_eq!(arr[0]["citations"]["enabled"], true);
    }
}

pub fn convert_message_content(content: &MessageContent) -> Result<serde_json::Value, LlmError> {
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
                            crate::types::chat::MediaSource::Base64 { data } => {
                                let bytes = base64::engine::general_purpose::STANDARD
                                    .decode(data.as_bytes())
                                    .ok();
                                let media_type = bytes
                                    .as_deref()
                                    .map(super::headers::guess_image_media_type_from_bytes)
                                    .unwrap_or_else(|| "image/jpeg".to_string());
                                (media_type, data.clone())
                            }
                            crate::types::chat::MediaSource::Binary { data } => {
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                let media_type =
                                    super::headers::guess_image_media_type_from_bytes(data);
                                (media_type, encoded)
                            }
                            crate::types::chat::MediaSource::Url { url } => {
                                // Anthropic doesn't support URLs, convert to text
                                content_parts.push(serde_json::json!({
                                    "type": "text",
                                    "text": format!("[Image: {}]", url)
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
                    ContentPart::Source {
                        source_type: _,
                        url,
                        title,
                        ..
                    } => {
                        // Anthropic does not support `source` parts in request input.
                        // Convert them into a best-effort text placeholder to preserve context.
                        let text = if !title.is_empty() && title != url {
                            format!("[Source: {title} ({url})]")
                        } else {
                            format!("[Source: {url}]")
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
                        provider_metadata,
                        ..
                    } => {
                        fn apply_anthropic_document_options(
                            doc: &mut serde_json::Value,
                            filename: &Option<String>,
                            provider_metadata: &Option<
                                std::collections::HashMap<String, serde_json::Value>,
                            >,
                        ) {
                            let Some(provider_metadata) = provider_metadata else {
                                if let Some(name) = filename.as_ref() {
                                    doc["title"] = serde_json::json!(name);
                                }
                                return;
                            };

                            let Some(anthropic) = provider_metadata.get("anthropic") else {
                                if let Some(name) = filename.as_ref() {
                                    doc["title"] = serde_json::json!(name);
                                }
                                return;
                            };

                            let Some(obj) = anthropic.as_object() else {
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
                                crate::types::chat::MediaSource::Base64 { data } => {
                                    serde_json::json!({
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": data
                                    })
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    serde_json::json!({
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": encoded
                                    })
                                }
                                crate::types::chat::MediaSource::Url { url } => {
                                    serde_json::json!({
                                        "type": "url",
                                        "url": url
                                    })
                                }
                            };

                            let mut doc = serde_json::json!({
                                "type": "document",
                                "source": source_json,
                            });
                            apply_anthropic_document_options(&mut doc, filename, provider_metadata);
                            content_parts.push(doc);
                        } else if media_type == "text/plain" {
                            let source_json = match source {
                                crate::types::chat::MediaSource::Url { url } => {
                                    serde_json::json!({
                                        "type": "url",
                                        "url": url
                                    })
                                }
                                crate::types::chat::MediaSource::Binary { data } => {
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
                                crate::types::chat::MediaSource::Base64 { data } => {
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
                            };

                            let mut doc = serde_json::json!({
                                "type": "document",
                                "source": source_json,
                            });
                            apply_anthropic_document_options(&mut doc, filename, provider_metadata);
                            content_parts.push(doc);
                        } else {
                            content_parts.push(serde_json::json!({
                                "type": "text",
                                "text": format!("[Unsupported file type: {}]", media_type)
                            }));
                        }
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
                            ToolResultOutput::Text { value } => (serde_json::json!(value), false),
                            // Anthropic Messages API requires tool_result `content` to be a string
                            // or an array of content blocks; JSON objects must be stringified.
                            ToolResultOutput::Json { value } => (
                                serde_json::json!(serde_json::to_string(value).unwrap_or_default()),
                                false,
                            ),
                            ToolResultOutput::ErrorText { value } => {
                                (serde_json::json!(value), true)
                            }
                            ToolResultOutput::ErrorJson { value } => (
                                serde_json::json!(serde_json::to_string(value).unwrap_or_default()),
                                true,
                            ),
                            ToolResultOutput::ExecutionDenied { reason } => {
                                let msg = reason
                                    .as_ref()
                                    .map(|r| format!("Execution denied: {}", r))
                                    .unwrap_or_else(|| "Execution denied".to_string());
                                (serde_json::json!(msg), true)
                            }
                            ToolResultOutput::Content { value } => {
                                // Convert multimodal content to Anthropic format
                                let content_array: Vec<serde_json::Value> = value.iter().map(|part| {
                                    use crate::types::ToolResultContentPart;
                                    match part {
                                        ToolResultContentPart::Text { text } => {
                                            serde_json::json!({"type": "text", "text": text})
                                        }
                                        ToolResultContentPart::Image { source, .. } => {
                                            use crate::types::MediaSource;
                                            match source {
                                                MediaSource::Url { url } => {
                                                    serde_json::json!({
                                                        "type": "text",
                                                        "text": format!("[Image: {}]", url)
                                                    })
                                                }
                                                MediaSource::Base64 { data } => {
                                                    let bytes = base64::engine::general_purpose::STANDARD
                                                        .decode(data.as_bytes())
                                                        .ok();
                                                    let media_type = bytes
                                                        .as_deref()
                                                        .map(super::headers::guess_image_media_type_from_bytes)
                                                        .unwrap_or_else(|| "image/jpeg".to_string());
                                                    serde_json::json!({
                                                        "type": "image",
                                                        "source": {
                                                            "type": "base64",
                                                            "media_type": media_type,
                                                            "data": data
                                                        }
                                                    })
                                                }
                                                MediaSource::Binary { data } => {
                                                    let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                                                    let media_type = super::headers::guess_image_media_type_from_bytes(data);
                                                    serde_json::json!({
                                                        "type": "image",
                                                        "source": {
                                                            "type": "base64",
                                                            "media_type": media_type,
                                                            "data": encoded
                                                        }
                                                    })
                                                }
                                            }
                                        }
                                        ToolResultContentPart::File { .. } => {
                                            serde_json::json!({"type": "text", "text": "[File attachment]"})
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
                        // Emit as a thinking block (Anthropic format). If the caller does not provide
                        // a valid signature (required for replaying thinking blocks), we will
                        // degrade it to a plain text `<thinking>...</thinking>` wrapper later in
                        // `convert_messages`.
                        content_parts.push(serde_json::json!({
                            "type": "thinking",
                            "thinking": text
                        }));
                    }
                }
            }

            Ok(serde_json::Value::Array(content_parts))
        }
        #[cfg(feature = "structured-messages")]
        MessageContent::Json(v) => {
            let s = serde_json::to_string(v).unwrap_or_default();
            Ok(serde_json::Value::Array(vec![serde_json::json!({
                "type": "text",
                "text": s
            })]))
        }
    }
}
