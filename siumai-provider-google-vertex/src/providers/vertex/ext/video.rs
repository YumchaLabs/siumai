use crate::provider_options::vertex::GoogleVertexVideoModelOptions;
use crate::types::video::VideoGenerationRequest;

fn upsert_vertex(map: &mut crate::types::ProviderOptionsMap, value: serde_json::Value) {
    if let serde_json::Value::Object(new_options) = value {
        let mut merged = map
            .get("vertex")
            .and_then(|value| value.as_object())
            .cloned()
            .unwrap_or_default();

        for (key, value) in new_options {
            merged.insert(key, value);
        }

        map.insert("vertex", serde_json::Value::Object(merged));
    } else {
        map.insert("vertex", value);
    }
}

/// Google Vertex video request helpers for AI SDK-aligned video request families.
pub trait VertexVideoRequestExt {
    /// Convenience: attach Google Vertex video options to `provider_options_map["vertex"]`.
    fn with_vertex_video_options(self, options: GoogleVertexVideoModelOptions) -> Self;
}

impl VertexVideoRequestExt for VideoGenerationRequest {
    fn with_vertex_video_options(mut self, options: GoogleVertexVideoModelOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize GoogleVertexVideoModelOptions");
        upsert_vertex(&mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider_options::vertex::GoogleVertexReferenceImage;

    #[test]
    fn vertex_video_request_ext_merges_existing_vertex_video_options() {
        let request = VideoGenerationRequest::new("veo-3.1-generate-preview", "hi")
            .with_provider_option(
                "vertex",
                serde_json::json!({
                    "existing": true,
                    "negativePrompt": "old"
                }),
            )
            .with_vertex_video_options(
                GoogleVertexVideoModelOptions::new()
                    .with_negative_prompt("new negative prompt")
                    .with_reference_images(vec![
                        GoogleVertexReferenceImage::new().with_gcs_uri("gs://bucket/reference.png"),
                    ]),
            );

        let value = request
            .provider_options_map
            .get("vertex")
            .expect("vertex options present");
        assert_eq!(value["existing"], serde_json::json!(true));
        assert_eq!(
            value["negativePrompt"],
            serde_json::json!("new negative prompt")
        );
        assert_eq!(
            value["referenceImages"][0]["gcsUri"],
            serde_json::json!("gs://bucket/reference.png")
        );
    }
}
