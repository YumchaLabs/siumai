use crate::provider_options::vertex::VertexImagenOptions;

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

pub trait VertexImagenRequestExt {
    fn with_vertex_imagen_options(self, options: VertexImagenOptions) -> Self;
}

impl VertexImagenRequestExt for crate::types::ImageGenerationRequest {
    fn with_vertex_imagen_options(mut self, options: VertexImagenOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize VertexImagenOptions");
        upsert_vertex(&mut self.provider_options_map, value);
        self
    }
}

impl VertexImagenRequestExt for crate::types::ImageEditRequest {
    fn with_vertex_imagen_options(mut self, options: VertexImagenOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize VertexImagenOptions");
        upsert_vertex(&mut self.provider_options_map, value);
        self
    }
}

impl VertexImagenRequestExt for crate::types::ImageVariationRequest {
    fn with_vertex_imagen_options(mut self, options: VertexImagenOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize VertexImagenOptions");
        upsert_vertex(&mut self.provider_options_map, value);
        self
    }
}

impl VertexImagenRequestExt for crate::types::GenerateImageRequest {
    fn with_vertex_imagen_options(mut self, options: VertexImagenOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize VertexImagenOptions");
        upsert_vertex(&mut self.provider_options_map, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_image_request_ext_merges_existing_vertex_imagen_options() {
        let request = crate::types::GenerateImageRequest::new("draw a robot")
            .with_provider_option(
                "vertex",
                serde_json::json!({
                    "existing": true,
                    "negativePrompt": "old"
                }),
            )
            .with_vertex_imagen_options(
                VertexImagenOptions::new().with_negative_prompt("new negative prompt"),
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
    }
}
