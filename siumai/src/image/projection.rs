use base64::{Engine, engine::general_purpose::STANDARD};
use siumai_core::error::LlmError;
use siumai_core::types::{
    GenerateImageResult, GeneratedFile, GeneratedImage, HttpResponseInfo, ImageGenerationResponse,
    ImageModelProviderMetadata, ImageModelResponseMetadata, ImageModelUsage, JSONValue,
    merge_provider_metadata, provider_metadata_from_object,
};
use siumai_provider_utils::download_url;
use siumai_provider_utils::mime::{guess_mime_from_bytes, guess_mime_from_path_or_url};
use std::collections::HashMap;
fn normalize_image_media_type(value: &str) -> Option<String> {
    let media_type = value.split(';').next()?.trim();
    if media_type.is_empty() {
        return None;
    }
    Some(match media_type.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg".to_string(),
        "png" => "image/png".to_string(),
        "webp" => "image/webp".to_string(),
        "gif" => "image/gif".to_string(),
        "bmp" => "image/bmp".to_string(),
        "svg" => "image/svg+xml".to_string(),
        value if value.starts_with("image/") => value.to_string(),
        value if !value.contains('/') => format!("image/{value}"),
        value => value.to_string(),
    })
}

fn image_metadata_media_type(metadata: &HashMap<String, JSONValue>) -> Option<String> {
    metadata
        .get("mediaType")
        .or_else(|| metadata.get("media_type"))
        .or_else(|| metadata.get("mimeType"))
        .or_else(|| metadata.get("mime_type"))
        .and_then(JSONValue::as_str)
        .and_then(normalize_image_media_type)
}

fn generated_image_media_type(
    image: &GeneratedImage,
    bytes: Option<&[u8]>,
    downloaded_media_type: Option<&str>,
) -> String {
    image_metadata_media_type(&image.metadata)
        .or_else(|| downloaded_media_type.and_then(normalize_image_media_type))
        .or_else(|| bytes.and_then(guess_mime_from_bytes))
        .or_else(|| image.format.as_deref().and_then(normalize_image_media_type))
        .or_else(|| image.url.as_deref().and_then(guess_mime_from_path_or_url))
        .unwrap_or_else(|| "image/png".to_string())
}

async fn generated_image_to_file(image: &GeneratedImage) -> Result<GeneratedFile, LlmError> {
    if let Some(base64) = image.b64_json.as_deref() {
        if base64.starts_with("data:") {
            let downloaded = download_url(base64)
                .await
                .map_err(|error| LlmError::HttpError(error.message))?;
            let media_type = generated_image_media_type(
                image,
                Some(downloaded.data.as_slice()),
                downloaded.media_type.as_deref(),
            );
            return Ok(GeneratedFile::from_bytes(downloaded.data, media_type));
        }

        let bytes = STANDARD.decode(base64).map_err(|error| {
            LlmError::InvalidInput(format!("Invalid generated image base64 data: {error}"))
        })?;
        let media_type = generated_image_media_type(image, Some(bytes.as_slice()), None);
        return Ok(GeneratedFile::from_base64(base64.to_string(), media_type));
    }

    let Some(url) = image.url.as_deref() else {
        return Err(LlmError::ParseError(
            "Generated image was missing both `b64_json` and `url`".to_string(),
        ));
    };

    let downloaded = download_url(url)
        .await
        .map_err(|error| LlmError::HttpError(error.message))?;
    let media_type = generated_image_media_type(
        image,
        Some(downloaded.data.as_slice()),
        downloaded.media_type.as_deref(),
    );
    Ok(GeneratedFile::from_bytes(downloaded.data, media_type))
}

fn merge_image_provider_metadata(
    target: &mut ImageModelProviderMetadata,
    source: ImageModelProviderMetadata,
) {
    for (provider_id, source_value) in source {
        match (target.get_mut(&provider_id), source_value) {
            (Some(JSONValue::Object(target_obj)), JSONValue::Object(mut source_obj)) => {
                if let Some(JSONValue::Array(mut source_images)) = source_obj.remove("images") {
                    match target_obj.get_mut("images") {
                        Some(JSONValue::Array(target_images)) => {
                            target_images.append(&mut source_images);
                        }
                        _ => {
                            target_obj
                                .insert("images".to_string(), JSONValue::Array(source_images));
                        }
                    }
                }
                target_obj.extend(source_obj);
            }
            (_, value) => {
                target.insert(provider_id, value);
            }
        }
    }
}

fn provider_metadata_from_image_metadata_entry(
    provider_id: &str,
    mut metadata: HashMap<String, JSONValue>,
) -> ImageModelProviderMetadata {
    let mut provider_metadata = ImageModelProviderMetadata::default();

    if let Some(value) = metadata.remove(provider_id) {
        merge_provider_metadata(
            &mut provider_metadata,
            HashMap::from([(provider_id.to_string(), value)]),
        );
    }

    if !metadata.is_empty() {
        merge_provider_metadata(
            &mut provider_metadata,
            provider_metadata_from_object(provider_id.to_string(), metadata),
        );
    }

    provider_metadata
}

fn provider_metadata_from_image_metadata(
    provider_id: &str,
    metadata: &HashMap<String, JSONValue>,
) -> ImageModelProviderMetadata {
    let mut provider_metadata = ImageModelProviderMetadata::default();

    if let Some(entries) = metadata
        .get("_siumai")
        .and_then(|value| value.get("metadata"))
        .and_then(JSONValue::as_array)
    {
        for entry in entries {
            if let Ok(entry_metadata) =
                serde_json::from_value::<HashMap<String, JSONValue>>(entry.clone())
            {
                merge_image_provider_metadata(
                    &mut provider_metadata,
                    provider_metadata_from_image_metadata_entry(provider_id, entry_metadata),
                );
            }
        }
    }

    merge_image_provider_metadata(
        &mut provider_metadata,
        provider_metadata_from_image_metadata_entry(provider_id, metadata.clone()),
    );

    provider_metadata
}

fn u32_metadata_value(value: Option<&JSONValue>) -> Option<u32> {
    value
        .and_then(JSONValue::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn image_usage_from_metadata_entry(metadata: &HashMap<String, JSONValue>) -> ImageModelUsage {
    let usage = metadata
        .get("usage")
        .and_then(JSONValue::as_object)
        .or_else(|| {
            metadata.values().find_map(|value| {
                value
                    .as_object()
                    .and_then(|object| object.get("usage"))
                    .and_then(JSONValue::as_object)
            })
        });

    let Some(usage) = usage else {
        return ImageModelUsage::default();
    };

    ImageModelUsage::new(
        u32_metadata_value(
            usage
                .get("inputTokens")
                .or_else(|| usage.get("input_tokens")),
        ),
        u32_metadata_value(
            usage
                .get("outputTokens")
                .or_else(|| usage.get("output_tokens")),
        ),
        u32_metadata_value(
            usage
                .get("totalTokens")
                .or_else(|| usage.get("total_tokens")),
        ),
    )
}

fn image_usage_from_metadata(metadata: &HashMap<String, JSONValue>) -> ImageModelUsage {
    let mut usage = ImageModelUsage::default();

    if let Some(entries) = metadata
        .get("_siumai")
        .and_then(|value| value.get("metadata"))
        .and_then(JSONValue::as_array)
    {
        for entry in entries {
            if let Ok(entry_metadata) =
                serde_json::from_value::<HashMap<String, JSONValue>>(entry.clone())
            {
                usage.merge(&image_usage_from_metadata_entry(&entry_metadata));
            }
        }
    }

    usage.merge(&image_usage_from_metadata_entry(metadata));
    usage
}

fn image_response_metadata_from_response(
    metadata: &HashMap<String, JSONValue>,
    response: Option<&HttpResponseInfo>,
) -> Vec<ImageModelResponseMetadata> {
    let mut responses = Vec::new();

    if let Some(entries) = metadata
        .get("_siumai")
        .and_then(|value| value.get("responses"))
        .and_then(JSONValue::as_array)
    {
        responses.extend(entries.iter().filter_map(|entry| {
            serde_json::from_value::<HttpResponseInfo>(entry.clone())
                .ok()
                .and_then(|response| ImageModelResponseMetadata::try_from(response).ok())
        }));
    }

    if responses.is_empty()
        && let Some(response) = response
        && let Ok(metadata) = ImageModelResponseMetadata::try_from(response)
    {
        responses.push(metadata);
    }

    responses
}

pub(super) async fn project_generate_image_response(
    provider_id: &str,
    response: ImageGenerationResponse,
) -> Result<GenerateImageResult, LlmError> {
    let ImageGenerationResponse {
        images,
        metadata,
        warnings,
        response,
    } = response;

    let mut files = Vec::with_capacity(images.len());
    for image in &images {
        files.push(generated_image_to_file(image).await?);
    }

    let Some(result) = GenerateImageResult::from_images(files) else {
        return Err(LlmError::NoImageGenerated {
            responses: response.into_iter().collect(),
        });
    };

    Ok(result
        .with_warnings(warnings.unwrap_or_default())
        .with_responses(image_response_metadata_from_response(
            &metadata,
            response.as_ref(),
        ))
        .with_provider_metadata(provider_metadata_from_image_metadata(
            provider_id,
            &metadata,
        ))
        .with_usage(image_usage_from_metadata(&metadata)))
}
