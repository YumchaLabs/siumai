use crate::provider_options::vertex::VertexEmbeddingOptions;

fn upsert_vertex(map: &mut crate::types::ProviderOptionsMap, value: serde_json::Value) {
    map.insert("vertex", value);
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

#[cfg(test)]
mod tests {
    fn source_section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
        let start_index = source.find(start).expect("section start marker");
        let end_index = source[start_index..]
            .find(end)
            .map(|offset| start_index + offset)
            .expect("section end marker");
        &source[start_index..end_index]
    }

    #[test]
    fn vertex_embedding_request_option_extension_source_does_not_read_response_metadata() {
        let source = include_str!("embedding.rs");
        let request_source = source_section(source, "use crate::provider_options", "#[cfg(test)]");

        for disallowed in ["provider_metadata", "ProviderMetadata", "ChatResponse"] {
            assert!(
                !request_source.contains(disallowed),
                "Vertex embedding request option extension helpers must stay request-only"
            );
        }
    }
}
