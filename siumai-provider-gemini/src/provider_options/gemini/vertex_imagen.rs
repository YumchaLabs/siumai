//! Vertex AI Imagen provider options (Gemini provider id).
//!
//! These options are serialized under:
//! - `providerOptions["gemini"]["vertexImagen"]`
//!
//! The actual wire format is passed through to the Vertex AI `:predict` endpoint.

use base64::Engine;
use serde::{Deserialize, Serialize};

fn guess_mime(bytes: &[u8]) -> &'static str {
    infer::get(bytes)
        .map(|t| t.mime_type())
        .unwrap_or("image/png")
}

/// Vertex AI inline image value used by Imagen requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexImagenInlineImage {
    /// Base64 image bytes (Vertex field: `bytesBase64Encoded`).
    #[serde(rename = "bytesBase64Encoded")]
    pub bytes_base64_encoded: String,
    /// Optional MIME type (Vertex field: `mimeType`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "mimeType")]
    pub mime_type: Option<String>,
}

impl VertexImagenInlineImage {
    /// Build from raw image bytes (MIME type is best-effort detected).
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Self {
        let bytes = bytes.as_ref();
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Self {
            bytes_base64_encoded: b64,
            mime_type: Some(guess_mime(bytes).to_string()),
        }
    }
}

/// Reference image entry for Vertex AI Imagen.
///
/// This is passed through as `referenceImages[]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexImagenReferenceImage {
    /// The image payload.
    #[serde(rename = "referenceImage")]
    pub reference_image: VertexImagenInlineImage,
    /// Reference type hint (provider-defined string).
    #[serde(skip_serializing_if = "Option::is_none", rename = "referenceType")]
    pub reference_type: Option<String>,
}

impl VertexImagenReferenceImage {
    /// Create a reference image from bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Self {
        Self {
            reference_image: VertexImagenInlineImage::from_bytes(bytes),
            reference_type: None,
        }
    }

    /// Set reference type hint.
    pub fn with_reference_type(mut self, reference_type: impl Into<String>) -> Self {
        self.reference_type = Some(reference_type.into());
        self
    }
}

/// Options for Vertex AI Imagen requests.
///
/// These are forwarded to the Vertex `:predict` payload. Fields here are a small
/// subset that commonly shows up in Vercel-aligned usage; additional parameters
/// can be passed via `extra_params`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexImagenOptions {
    /// Negative prompt hint.
    #[serde(skip_serializing_if = "Option::is_none", rename = "negativePrompt")]
    pub negative_prompt: Option<String>,
    /// Reference images for style/subject guidance.
    #[serde(skip_serializing_if = "Option::is_none", rename = "referenceImages")]
    pub reference_images: Option<Vec<VertexImagenReferenceImage>>,
}

impl VertexImagenOptions {
    /// Create empty options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set negative prompt.
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Set reference images.
    pub fn with_reference_images(
        mut self,
        reference_images: Vec<VertexImagenReferenceImage>,
    ) -> Self {
        self.reference_images = Some(reference_images);
        self
    }
}
