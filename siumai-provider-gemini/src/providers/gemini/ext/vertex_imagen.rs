use crate::provider_options::gemini::VertexImagenOptions;
use crate::types::{ImageEditRequest, ImageGenerationRequest};

fn upsert_vertex_imagen(
    provider_options_map: &mut crate::types::ProviderOptionsMap,
    value: serde_json::Value,
) {
    let entry = provider_options_map
        .0
        .entry("gemini".to_string())
        .or_insert_with(|| serde_json::json!({}));

    if !entry.is_object() {
        *entry = serde_json::json!({});
    }

    if let Some(obj) = entry.as_object_mut() {
        // Vercel alignment: use `providerOptions.gemini.vertex` for Vertex AI knobs.
        // Keep a legacy-compatible key reader (`vertexImagen`) on the transformer side.
        obj.insert("vertex".to_string(), value);
    }
}

/// Vertex AI Imagen request option helpers.
///
/// These helpers attach typed options under `providerOptions["gemini"]["vertex"]`.
pub trait VertexImagenRequestExt {
    /// Attach Vertex Imagen options to the request.
    fn with_vertex_imagen_options(self, options: VertexImagenOptions) -> Self;
}

impl VertexImagenRequestExt for ImageGenerationRequest {
    fn with_vertex_imagen_options(mut self, options: VertexImagenOptions) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        upsert_vertex_imagen(&mut self.provider_options_map, value);
        self
    }
}

impl VertexImagenRequestExt for ImageEditRequest {
    fn with_vertex_imagen_options(mut self, options: VertexImagenOptions) -> Self {
        let value = serde_json::to_value(options).unwrap_or(serde_json::Value::Null);
        upsert_vertex_imagen(&mut self.provider_options_map, value);
        self
    }
}
