//! OpenAI Responses provider metadata, source, and logprob aggregation.

use std::collections::{HashMap, HashSet};

use crate::standards::openai::compat::usage::xai_responses_usage_provider_metadata_value;

use super::ResponsesTransformStyle;

pub(super) fn response_provider_metadata(
    root: &serde_json::Value,
    style: ResponsesTransformStyle,
    provider_metadata_key: &str,
) -> Option<HashMap<String, serde_json::Value>> {
    match style {
        ResponsesTransformStyle::Xai => root
            .get("usage")
            .and_then(xai_responses_usage_provider_metadata_value)
            .map(|metadata| single_provider_metadata_map(provider_metadata_key, metadata)),
        ResponsesTransformStyle::OpenAi => openai_response_provider_metadata(
            root,
            provider_metadata_key,
            collect_openai_sources(root, provider_metadata_key),
        ),
    }
}

fn openai_response_provider_metadata(
    root: &serde_json::Value,
    provider_metadata_key: &str,
    sources: Vec<serde_json::Value>,
) -> Option<HashMap<String, serde_json::Value>> {
    let mut openai_meta = serde_json::Map::new();

    if let Some(response_id) = root.get("id").and_then(|v| v.as_str()) {
        openai_meta.insert(
            "responseId".to_string(),
            serde_json::Value::String(response_id.to_string()),
        );
    }

    if let Some(service_tier) = root.get("service_tier").and_then(|v| v.as_str()) {
        openai_meta.insert(
            "serviceTier".to_string(),
            serde_json::Value::String(service_tier.to_string()),
        );
    }

    if !sources.is_empty() {
        openai_meta.insert("sources".to_string(), serde_json::Value::Array(sources));
    }

    if let Some(logprobs) = output_text_logprobs(root) {
        openai_meta.insert("logprobs".to_string(), logprobs);
    }

    if openai_meta.is_empty() {
        None
    } else {
        Some(single_provider_metadata_map(
            provider_metadata_key,
            serde_json::Value::Object(openai_meta),
        ))
    }
}

fn collect_openai_sources(
    root: &serde_json::Value,
    provider_metadata_key: &str,
) -> Vec<serde_json::Value> {
    let mut sources: Vec<serde_json::Value> = Vec::new();
    let mut seen_source_keys: HashSet<String> = HashSet::new();

    let Some(output) = root.get("output").and_then(|v| v.as_array()) else {
        return sources;
    };

    collect_tool_result_sources(
        output,
        &mut sources,
        &mut seen_source_keys,
        provider_metadata_key,
    );
    collect_message_annotation_sources(
        output,
        &mut sources,
        &mut seen_source_keys,
        provider_metadata_key,
    );

    sources
}

fn collect_tool_result_sources(
    output: &[serde_json::Value],
    sources: &mut Vec<serde_json::Value>,
    seen_source_keys: &mut HashSet<String>,
    provider_metadata_key: &str,
) {
    for item in output {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if !matches!(item_type, "web_search_call" | "file_search_call") {
            continue;
        }

        let tool_call_id = item
            .get("call_id")
            .and_then(|v| v.as_str())
            .or_else(|| item.get("id").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();
        if tool_call_id.is_empty() {
            continue;
        }

        let Some(results) = item.get("results").and_then(|v| v.as_array()) else {
            continue;
        };

        for (index, result) in results.iter().enumerate() {
            let Some(result) = result.as_object() else {
                continue;
            };

            if item_type == "web_search_call" {
                collect_web_search_source(index, result, &tool_call_id, sources, seen_source_keys);
            } else {
                collect_file_search_source(
                    index,
                    result,
                    &tool_call_id,
                    sources,
                    seen_source_keys,
                    provider_metadata_key,
                );
            }
        }
    }
}

fn collect_web_search_source(
    index: usize,
    result: &serde_json::Map<String, serde_json::Value>,
    tool_call_id: &str,
    sources: &mut Vec<serde_json::Value>,
    seen_source_keys: &mut HashSet<String>,
) {
    let url = result.get("url").and_then(|v| v.as_str()).unwrap_or("");
    if url.is_empty() {
        return;
    }

    let source_key = format!("tool:{tool_call_id}:url:{url}");
    if !seen_source_keys.insert(source_key) {
        return;
    }

    let title = result
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let snippet = result
        .get("snippet")
        .or_else(|| result.get("text"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let source_id = result
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{tool_call_id}:{index}"));

    let mut source = serde_json::Map::new();
    source.insert("id".to_string(), serde_json::Value::String(source_id));
    source.insert(
        "source_type".to_string(),
        serde_json::Value::String("url".to_string()),
    );
    source.insert(
        "url".to_string(),
        serde_json::Value::String(url.to_string()),
    );
    if let Some(title) = title {
        source.insert("title".to_string(), serde_json::Value::String(title));
    }
    source.insert(
        "tool_call_id".to_string(),
        serde_json::Value::String(tool_call_id.to_string()),
    );
    if let Some(snippet) = snippet {
        source.insert("snippet".to_string(), serde_json::Value::String(snippet));
    }
    sources.push(serde_json::Value::Object(source));
}

fn collect_file_search_source(
    index: usize,
    result: &serde_json::Map<String, serde_json::Value>,
    tool_call_id: &str,
    sources: &mut Vec<serde_json::Value>,
    seen_source_keys: &mut HashSet<String>,
    provider_metadata_key: &str,
) {
    let file_id = result
        .get("file_id")
        .or_else(|| result.get("fileId"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if file_id.is_empty() {
        return;
    }

    let container_id = result
        .get("container_id")
        .or_else(|| result.get("containerId"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let result_index = result
        .get("index")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);
    let source_key = format!(
        "tool:{tool_call_id}:document:{file_id}:{}:{}",
        container_id.as_deref().unwrap_or(""),
        result_index.map(|v| v.to_string()).unwrap_or_default()
    );
    if !seen_source_keys.insert(source_key) {
        return;
    }

    let source_id = result
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{tool_call_id}:{index}"));
    let title = result
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let snippet = result
        .get("snippet")
        .or_else(|| result.get("text"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let filename = result
        .get("filename")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let media_type = result
        .get("media_type")
        .or_else(|| result.get("mediaType"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let mut openai_source_meta = serde_json::Map::new();
    openai_source_meta.insert(
        "fileId".to_string(),
        serde_json::Value::String(file_id.clone()),
    );
    if let Some(container_id) = &container_id {
        openai_source_meta.insert(
            "containerId".to_string(),
            serde_json::Value::String(container_id.clone()),
        );
    }
    if let Some(result_index) = result_index {
        openai_source_meta.insert("index".to_string(), serde_json::json!(result_index));
    }

    let mut source = serde_json::Map::new();
    source.insert("id".to_string(), serde_json::Value::String(source_id));
    source.insert(
        "source_type".to_string(),
        serde_json::Value::String("document".to_string()),
    );
    source.insert("url".to_string(), serde_json::Value::String(file_id));
    source.insert(
        "tool_call_id".to_string(),
        serde_json::Value::String(tool_call_id.to_string()),
    );
    if let Some(title) = title {
        source.insert("title".to_string(), serde_json::Value::String(title));
    }
    if let Some(snippet) = snippet {
        source.insert("snippet".to_string(), serde_json::Value::String(snippet));
    }
    if let Some(filename) = filename {
        source.insert("filename".to_string(), serde_json::Value::String(filename));
    }
    if let Some(media_type) = media_type {
        source.insert(
            "media_type".to_string(),
            serde_json::Value::String(media_type),
        );
    }
    source.insert(
        "provider_metadata".to_string(),
        single_provider_metadata_value(
            provider_metadata_key,
            serde_json::Value::Object(openai_source_meta),
        ),
    );
    sources.push(serde_json::Value::Object(source));
}

fn collect_message_annotation_sources(
    output: &[serde_json::Value],
    sources: &mut Vec<serde_json::Value>,
    seen_source_keys: &mut HashSet<String>,
    provider_metadata_key: &str,
) {
    let mut annotation_index: u64 = 0;

    for item in output {
        if item.get("type").and_then(|v| v.as_str()) != Some("message") {
            continue;
        }
        let Some(content_parts) = item.get("content").and_then(|v| v.as_array()) else {
            continue;
        };

        for content_part in content_parts {
            let Some(annotations) = content_part.get("annotations").and_then(|v| v.as_array())
            else {
                continue;
            };

            for annotation in annotations {
                let annotation_type = annotation
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if annotation_type == "url_citation" {
                    if collect_url_citation_source(
                        annotation,
                        annotation_index,
                        sources,
                        seen_source_keys,
                    ) {
                        annotation_index += 1;
                    }
                    continue;
                }

                if matches!(
                    annotation_type,
                    "file_citation" | "container_file_citation" | "file_path"
                ) && collect_document_annotation_source(
                    annotation_type,
                    annotation,
                    annotation_index,
                    sources,
                    seen_source_keys,
                    provider_metadata_key,
                ) {
                    annotation_index += 1;
                }
            }
        }
    }
}

fn collect_url_citation_source(
    annotation: &serde_json::Value,
    annotation_index: u64,
    sources: &mut Vec<serde_json::Value>,
    seen_source_keys: &mut HashSet<String>,
) -> bool {
    let url = annotation.get("url").and_then(|v| v.as_str()).unwrap_or("");
    if url.is_empty() {
        return false;
    }

    let source_key = format!("message:url:{url}");
    if !seen_source_keys.insert(source_key) {
        return false;
    }

    let title = annotation
        .get("title")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let mut source = serde_json::Map::new();
    source.insert(
        "id".to_string(),
        serde_json::Value::String(format!("ann:{annotation_index}")),
    );
    source.insert(
        "source_type".to_string(),
        serde_json::Value::String("url".to_string()),
    );
    source.insert(
        "url".to_string(),
        serde_json::Value::String(url.to_string()),
    );
    if let Some(title) = title {
        source.insert("title".to_string(), serde_json::Value::String(title));
    }
    sources.push(serde_json::Value::Object(source));

    true
}

fn collect_document_annotation_source(
    annotation_type: &str,
    annotation: &serde_json::Value,
    annotation_index: u64,
    sources: &mut Vec<serde_json::Value>,
    seen_source_keys: &mut HashSet<String>,
    provider_metadata_key: &str,
) -> bool {
    let file_id = annotation
        .get("file_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if file_id.is_empty() {
        return false;
    }

    let filename = annotation
        .get("filename")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| Some(file_id.clone()));

    let title = annotation
        .get("quote")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| filename.clone())
        .or_else(|| Some("Document".to_string()));

    let media_type = if annotation_type == "file_path" {
        Some("application/octet-stream".to_string())
    } else {
        Some("text/plain".to_string())
    };
    let index = annotation.get("index").and_then(|v| v.as_u64());
    let container_id = annotation
        .get("container_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let source_key = format!(
        "message:doc:{annotation_type}:{file_id}:{}:{}:{}:{}",
        container_id.as_deref().unwrap_or(""),
        index.map(|value| value.to_string()).unwrap_or_default(),
        filename.as_deref().unwrap_or(""),
        title.as_deref().unwrap_or(""),
    );
    if !seen_source_keys.insert(source_key) {
        return false;
    }

    let provider_metadata = match annotation_type {
        "file_citation" => Some(single_provider_metadata_value(
            provider_metadata_key,
            serde_json::json!({
                "type": "file_citation",
                "fileId": file_id,
                "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
            }),
        )),
        "container_file_citation" => Some(single_provider_metadata_value(
            provider_metadata_key,
            serde_json::json!({
                "type": "container_file_citation",
                "fileId": file_id,
                "containerId": annotation.get("container_id").cloned().unwrap_or(serde_json::Value::Null),
                "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
            }),
        )),
        "file_path" => Some(single_provider_metadata_value(
            provider_metadata_key,
            serde_json::json!({
                "type": "file_path",
                "fileId": file_id,
                "index": annotation.get("index").cloned().unwrap_or(serde_json::Value::Null),
            }),
        )),
        _ => None,
    };

    let mut source = serde_json::Map::new();
    source.insert(
        "id".to_string(),
        serde_json::Value::String(format!("ann:{annotation_index}")),
    );
    source.insert(
        "source_type".to_string(),
        serde_json::Value::String("document".to_string()),
    );
    source.insert("url".to_string(), serde_json::Value::String(file_id));
    if let Some(title) = title {
        source.insert("title".to_string(), serde_json::Value::String(title));
    }
    if let Some(media_type) = media_type {
        source.insert(
            "media_type".to_string(),
            serde_json::Value::String(media_type),
        );
    }
    if let Some(filename) = filename {
        source.insert("filename".to_string(), serde_json::Value::String(filename));
    }
    if let Some(provider_metadata) = provider_metadata {
        source.insert("provider_metadata".to_string(), provider_metadata);
    }
    sources.push(serde_json::Value::Object(source));

    true
}

pub(crate) fn output_text_logprobs(root: &serde_json::Value) -> Option<serde_json::Value> {
    let output = root.get("output")?.as_array()?;

    let mut outer: Vec<serde_json::Value> = Vec::new();
    for item in output {
        if item.get("type").and_then(|v| v.as_str()) != Some("message") {
            continue;
        }

        let content = item.get("content").and_then(|v| v.as_array());
        let Some(content) = content else { continue };

        for part in content {
            if part.get("type").and_then(|v| v.as_str()) != Some("output_text") {
                continue;
            }

            let logprobs = part.get("logprobs").and_then(|v| v.as_array());
            let Some(logprobs) = logprobs else { continue };

            let mut inner: Vec<serde_json::Value> = Vec::new();
            for entry in logprobs {
                let token = entry.get("token").and_then(|v| v.as_str()).unwrap_or("");
                if token.is_empty() {
                    continue;
                }

                let logprob = entry
                    .get("logprob")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                let mut out_entry = serde_json::Map::new();
                out_entry.insert(
                    "token".to_string(),
                    serde_json::Value::String(token.to_string()),
                );
                out_entry.insert("logprob".to_string(), logprob);

                let top = entry.get("top_logprobs").and_then(|v| v.as_array());
                if let Some(top) = top {
                    let mut tops: Vec<serde_json::Value> = Vec::new();
                    for top_entry in top {
                        let token = top_entry
                            .get("token")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if token.is_empty() {
                            continue;
                        }
                        let logprob = top_entry
                            .get("logprob")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        tops.push(serde_json::json!({
                            "token": token,
                            "logprob": logprob,
                        }));
                    }
                    out_entry.insert("top_logprobs".to_string(), serde_json::Value::Array(tops));
                } else {
                    out_entry.insert("top_logprobs".to_string(), serde_json::Value::Array(vec![]));
                }

                inner.push(serde_json::Value::Object(out_entry));
            }

            if !inner.is_empty() {
                outer.push(serde_json::Value::Array(inner));
            }
        }
    }

    if outer.is_empty() {
        None
    } else {
        Some(serde_json::Value::Array(outer))
    }
}

fn single_provider_metadata_map(
    provider_metadata_key: &str,
    value: serde_json::Value,
) -> HashMap<String, serde_json::Value> {
    let mut out = HashMap::new();
    out.insert(provider_metadata_key.to_string(), value);
    out
}

fn single_provider_metadata_value(
    provider_metadata_key: &str,
    value: serde_json::Value,
) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    out.insert(provider_metadata_key.to_string(), value);
    serde_json::Value::Object(out)
}
