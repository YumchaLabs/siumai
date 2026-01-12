use crate::provider_options::vertex::VertexImagenOptions;

fn upsert_vertex(map: &mut crate::types::ProviderOptionsMap, value: serde_json::Value) {
    map.insert("vertex", value);
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
