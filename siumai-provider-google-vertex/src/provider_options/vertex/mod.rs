pub mod embedding;
pub mod imagen;
pub mod shared;
pub mod video;

pub use embedding::VertexEmbeddingOptions;
pub use imagen::{
    VertexImagenEditMode, VertexImagenEditOptions, VertexImagenInlineImage,
    VertexImagenMaskImageConfig, VertexImagenMaskMode, VertexImagenOptions,
    VertexImagenReferenceImage, VertexImagenSafetySetting, VertexImagenSampleImageSize,
};
pub use shared::VertexPersonGeneration;
#[allow(deprecated)]
pub use video::{
    GoogleVertexReferenceImage, GoogleVertexVideoModelId, GoogleVertexVideoModelOptions,
    GoogleVertexVideoProviderOptions,
};

/// AI SDK-style alias for Google Vertex embedding model options.
pub type GoogleVertexEmbeddingModelOptions = VertexEmbeddingOptions;

/// AI SDK-style alias for Google Vertex image model options.
pub type GoogleVertexImageModelOptions = VertexImagenOptions;

/// Deprecated AI SDK compatibility alias for Google Vertex image options.
#[deprecated(note = "Use `GoogleVertexImageModelOptions` instead.")]
pub type GoogleVertexImageProviderOptions = GoogleVertexImageModelOptions;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn google_vertex_video_options_serialize_to_ai_sdk_shape() {
        let value = serde_json::to_value(
            GoogleVertexVideoModelOptions::new()
                .with_poll_interval_ms(500)
                .with_poll_timeout_ms(30_000)
                .with_person_generation(VertexPersonGeneration::AllowAdult)
                .with_negative_prompt("blurry")
                .with_generate_audio(true)
                .with_gcs_output_directory("gs://bucket/output/")
                .with_reference_images(vec![
                    GoogleVertexReferenceImage::new().with_bytes_base64_encoded("Zm9v"),
                    GoogleVertexReferenceImage::new().with_gcs_uri("gs://bucket/reference.png"),
                ])
                .with_extra_field("customFlag", serde_json::json!(true)),
        )
        .expect("serialize GoogleVertexVideoModelOptions");

        assert_eq!(
            value,
            serde_json::json!({
                "pollIntervalMs": 500,
                "pollTimeoutMs": 30000,
                "personGeneration": "allow_adult",
                "negativePrompt": "blurry",
                "generateAudio": true,
                "gcsOutputDirectory": "gs://bucket/output/",
                "referenceImages": [
                    {
                        "bytesBase64Encoded": "Zm9v"
                    },
                    {
                        "gcsUri": "gs://bucket/reference.png"
                    }
                ],
                "customFlag": true
            })
        );
    }

    #[test]
    #[allow(deprecated)]
    fn google_vertex_video_aliases_resolve_to_same_type() {
        let options = GoogleVertexVideoModelOptions::new()
            .with_negative_prompt("no cats")
            .with_generate_audio(true);
        let deprecated_options: GoogleVertexVideoProviderOptions = options.clone();
        let model_id: GoogleVertexVideoModelId = "veo-3.1-generate-preview".to_string();

        assert_eq!(options.negative_prompt, deprecated_options.negative_prompt);
        assert_eq!(options.generate_audio, deprecated_options.generate_audio);
        assert_eq!(model_id, "veo-3.1-generate-preview");
    }

    #[test]
    fn typed_vertex_option_enums_serialize_to_ai_sdk_strings() {
        assert_eq!(
            serde_json::to_value(VertexPersonGeneration::AllowAll).unwrap(),
            serde_json::json!("allow_all")
        );
        assert_eq!(
            serde_json::to_value(VertexImagenSafetySetting::BlockMediumAndAbove).unwrap(),
            serde_json::json!("block_medium_and_above")
        );
        assert_eq!(
            serde_json::to_value(VertexImagenSampleImageSize::TwoK).unwrap(),
            serde_json::json!("2K")
        );
        assert_eq!(
            serde_json::to_value(VertexImagenEditMode::InpaintInsertion).unwrap(),
            serde_json::json!("EDIT_MODE_INPAINT_INSERTION")
        );
        assert_eq!(
            serde_json::to_value(VertexImagenMaskMode::UserProvided).unwrap(),
            serde_json::json!("MASK_MODE_USER_PROVIDED")
        );
    }
}
