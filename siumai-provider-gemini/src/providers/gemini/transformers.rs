//! Gemini transformers with provider-owned model policy enforcement.

use crate::error::LlmError;
use crate::execution::transformers::request::{ImageHttpBody, RequestTransformer};
use crate::types::{
    ChatRequest, EmbeddingRequest, ImageEditRequest, ImageGenerationRequest, ImageVariationRequest,
    ModerationRequest, RerankRequest,
};

pub use crate::standards::gemini::transformers::{
    GeminiFilesTransformer, GeminiResponseTransformer, GeminiStreamChunkTransformer,
};

use super::model_policy::sanitize_sampling_parameters;
use super::types::GeminiConfig;

/// Request transformer that applies provider-owned model policy after protocol encoding.
///
/// The protocol transformer remains responsible for the wire format. This wrapper owns facts that
/// change with Google's model catalog, such as model-specific parameter deprecations.
#[derive(Clone)]
pub struct GeminiRequestTransformer {
    pub config: GeminiConfig,
}

impl GeminiRequestTransformer {
    fn protocol_transformer(
        &self,
    ) -> crate::standards::gemini::transformers::GeminiRequestTransformer {
        crate::standards::gemini::transformers::GeminiRequestTransformer {
            config: self.config.clone(),
        }
    }
}

impl RequestTransformer for GeminiRequestTransformer {
    fn provider_id(&self) -> &str {
        "gemini"
    }

    fn transform_chat(&self, req: &ChatRequest) -> Result<serde_json::Value, LlmError> {
        let mut body = self.protocol_transformer().transform_chat(req)?;
        sanitize_sampling_parameters(&req.common_params.model, &mut body);
        Ok(body)
    }

    fn transform_embedding(&self, req: &EmbeddingRequest) -> Result<serde_json::Value, LlmError> {
        self.protocol_transformer().transform_embedding(req)
    }

    fn transform_image(&self, req: &ImageGenerationRequest) -> Result<serde_json::Value, LlmError> {
        let mut body = self.protocol_transformer().transform_image(req)?;
        let model_id = req.model.as_deref().unwrap_or(&self.config.model);
        sanitize_sampling_parameters(model_id, &mut body);
        Ok(body)
    }

    fn transform_image_edit(&self, req: &ImageEditRequest) -> Result<ImageHttpBody, LlmError> {
        self.protocol_transformer().transform_image_edit(req)
    }

    fn transform_image_variation(
        &self,
        req: &ImageVariationRequest,
    ) -> Result<ImageHttpBody, LlmError> {
        self.protocol_transformer().transform_image_variation(req)
    }

    fn transform_rerank(&self, req: &RerankRequest) -> Result<serde_json::Value, LlmError> {
        self.protocol_transformer().transform_rerank(req)
    }

    fn transform_moderation(&self, req: &ModerationRequest) -> Result<serde_json::Value, LlmError> {
        self.protocol_transformer().transform_moderation(req)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::gemini::model_constants::gemini_3;
    use crate::types::{ChatMessage, CommonParams};

    fn request(model: &str) -> ChatRequest {
        ChatRequest::new(vec![ChatMessage::user("hello").build()]).with_common_params(
            CommonParams {
                model: model.to_string(),
                temperature: Some(0.4),
                top_p: Some(0.8),
                top_k: Some(20.0),
                max_tokens: Some(512),
                ..Default::default()
            },
        )
    }

    #[test]
    fn generate_content_omits_deprecated_sampling_controls_for_gemini_3_6() {
        let transformer = GeminiRequestTransformer {
            config: GeminiConfig::new("test-key")
                .with_model(gemini_3::GEMINI_3_6_FLASH.to_string()),
        };

        let body = transformer
            .transform_chat(&request(gemini_3::GEMINI_3_6_FLASH))
            .expect("transform request");

        let generation = body["generationConfig"]
            .as_object()
            .expect("generation config");
        assert!(!generation.contains_key("temperature"));
        assert!(!generation.contains_key("topP"));
        assert!(!generation.contains_key("topK"));
        assert_eq!(
            generation.get("maxOutputTokens"),
            Some(&serde_json::json!(512))
        );
    }

    #[test]
    fn generate_content_omits_config_and_request_sampling_for_gemini_3_5_flash_lite() {
        let config = GeminiConfig::new("test-key")
            .with_model(gemini_3::GEMINI_3_5_FLASH_LITE.to_string())
            .with_temperature(0.1)
            .with_top_p(0.2)
            .with_top_k(3);
        let transformer = GeminiRequestTransformer { config };

        let body = transformer
            .transform_chat(&request(gemini_3::GEMINI_3_5_FLASH_LITE))
            .expect("transform request");

        let generation = body["generationConfig"]
            .as_object()
            .expect("generation config");
        assert!(!generation.contains_key("temperature"));
        assert!(!generation.contains_key("topP"));
        assert!(!generation.contains_key("topK"));
    }

    #[test]
    fn generate_content_preserves_sampling_for_models_that_support_it() {
        let transformer = GeminiRequestTransformer {
            config: GeminiConfig::new("test-key")
                .with_model(gemini_3::GEMINI_3_5_FLASH.to_string()),
        };

        let body = transformer
            .transform_chat(&request(gemini_3::GEMINI_3_5_FLASH))
            .expect("transform request");

        assert_eq!(body["generationConfig"]["temperature"], 0.4);
        assert_eq!(body["generationConfig"]["topP"], 0.8);
        assert_eq!(body["generationConfig"]["topK"], 20);
    }
}
