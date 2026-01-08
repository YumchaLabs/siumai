use crate::provider_options::vertex::VertexEmbeddingOptions;

fn upsert_vertex(map: &mut crate::types::ProviderOptionsMap, value: serde_json::Value) {
    map.insert("vertex".to_string(), value);
}

pub trait VertexEmbeddingRequestExt {
    fn with_vertex_embedding_options(self, options: VertexEmbeddingOptions) -> Self;
}

impl VertexEmbeddingRequestExt for crate::types::EmbeddingRequest {
    fn with_vertex_embedding_options(mut self, options: VertexEmbeddingOptions) -> Self {
        let value = serde_json::to_value(options).expect("serialize VertexEmbeddingOptions");
        upsert_vertex(&mut self.provider_options_map, value);
        self
    }
}
