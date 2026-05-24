use crate::video::materialization::{provider_metadata_root_object, provider_metadata_video_value};
use crate::video::{GenerateVideoProviderMetadata, GenerateVideoResult, GeneratedVideo};
use siumai_core::error::LlmError;
use std::collections::HashMap;

pub(super) fn project_generate_video_result(
    result: GenerateVideoResult,
) -> Result<siumai_core::types::GenerateVideoResult, LlmError> {
    let mut files = Vec::with_capacity(result.videos.len());
    for video in &result.videos {
        files.push(siumai_core::types::GeneratedFile::from_base64(
            video.base64()?,
            video.media_type.clone(),
        ));
    }

    let Some(projected) = siumai_core::types::GenerateVideoResult::from_videos(files) else {
        let responses = result
            .responses
            .iter()
            .filter_map(|response| {
                response
                    .query_response
                    .clone()
                    .or_else(|| response.create_response.clone())
            })
            .collect();
        return Err(LlmError::NoVideoGenerated { responses });
    };

    let responses = result.video_model_responses();
    Ok(projected
        .with_warnings(result.warnings)
        .with_responses(responses)
        .with_provider_metadata(result.provider_metadata))
}

pub(super) fn build_call_provider_metadata(
    provider_id: &str,
    task_entry: serde_json::Value,
    videos: &[GeneratedVideo],
    create_metadata: &HashMap<String, serde_json::Value>,
    query_metadata: &HashMap<String, serde_json::Value>,
) -> GenerateVideoProviderMetadata {
    let video_entries = videos
        .iter()
        .map(provider_metadata_video_value)
        .collect::<Vec<_>>();

    let mut provider_root = provider_metadata_root_object(create_metadata, provider_id)
        .cloned()
        .unwrap_or_default();
    if let Some(query_provider_root) =
        provider_metadata_root_object(query_metadata, provider_id).cloned()
    {
        merge_provider_metadata_object(&mut provider_root, query_provider_root);
    }
    provider_root.insert(
        "tasks".to_string(),
        serde_json::Value::Array(vec![task_entry]),
    );
    if !video_entries.is_empty() {
        provider_root.insert(
            "videos".to_string(),
            serde_json::Value::Array(video_entries),
        );
    }

    GenerateVideoProviderMetadata::from([(
        provider_id.to_string(),
        serde_json::Value::Object(provider_root),
    )])
}

fn merge_provider_metadata_object(
    existing: &mut serde_json::Map<String, serde_json::Value>,
    incoming: serde_json::Map<String, serde_json::Value>,
) {
    for (key, value) in incoming {
        match (existing.get_mut(&key), value) {
            (
                Some(serde_json::Value::Array(existing_items)),
                serde_json::Value::Array(mut incoming_items),
            ) if key == "videos" || key == "tasks" => {
                existing_items.append(&mut incoming_items);
            }
            (_, value) => {
                existing.insert(key, value);
            }
        }
    }
}

pub(super) fn merge_provider_metadata(
    target: &mut GenerateVideoProviderMetadata,
    incoming: GenerateVideoProviderMetadata,
) {
    for (provider_id, value) in incoming {
        match (target.get_mut(&provider_id), value) {
            (
                Some(serde_json::Value::Object(existing)),
                serde_json::Value::Object(incoming_object),
            ) => merge_provider_metadata_object(existing, incoming_object),
            (_, value) => {
                target.insert(provider_id, value);
            }
        }
    }
}
