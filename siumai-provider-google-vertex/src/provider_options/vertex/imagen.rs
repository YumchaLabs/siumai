//! Vertex AI Imagen provider options (Vercel-aligned).
//!
//! These options are serialized under:
//! - `providerOptions["vertex"]`
//!
//! The actual wire format is passed through to the Vertex AI `:predict` endpoint.

use base64::Engine;
use serde::{Deserialize, Serialize};

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
    /// Build from raw image bytes.
    ///
    /// Note: we intentionally omit `mimeType` by default to match the Vercel AI SDK
    /// request shape for Vertex Imagen.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Self {
        let bytes = bytes.as_ref();
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Self {
            bytes_base64_encoded: b64,
            mime_type: None,
        }
    }

    /// Set `mimeType` explicitly.
    pub fn with_mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }
}

/// Mask configuration used by Vertex Imagen editing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexImagenMaskImageConfig {
    /// Mask mode (e.g., `MASK_MODE_USER_PROVIDED`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "maskMode")]
    pub mask_mode: Option<String>,
    /// Optional dilation (percentage of image width), recommended `0.01`.
    #[serde(skip_serializing_if = "Option::is_none", rename = "dilation")]
    pub dilation: Option<f64>,
}

impl VertexImagenMaskImageConfig {
    pub fn user_provided() -> Self {
        Self {
            mask_mode: Some("MASK_MODE_USER_PROVIDED".to_string()),
            dilation: None,
        }
    }

    pub fn with_mask_mode(mut self, mode: impl Into<String>) -> Self {
        self.mask_mode = Some(mode.into());
        self
    }

    pub fn with_dilation(mut self, dilation: f64) -> Self {
        self.dilation = Some(dilation);
        self
    }
}

/// Reference image entry for Vertex AI Imagen.
///
/// This is passed through as `referenceImages[]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VertexImagenReferenceImage {
    /// Optional reference id (used by Vertex Imagen edit requests).
    #[serde(skip_serializing_if = "Option::is_none", rename = "referenceId")]
    pub reference_id: Option<u32>,
    /// The image payload.
    #[serde(rename = "referenceImage")]
    pub reference_image: VertexImagenInlineImage,
    /// Reference type hint (provider-defined string).
    #[serde(skip_serializing_if = "Option::is_none", rename = "referenceType")]
    pub reference_type: Option<String>,
    /// Optional mask config for mask reference images.
    #[serde(skip_serializing_if = "Option::is_none", rename = "maskImageConfig")]
    pub mask_image_config: Option<VertexImagenMaskImageConfig>,
}

impl VertexImagenReferenceImage {
    /// Create a reference image from bytes.
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Self {
        Self {
            reference_id: None,
            reference_image: VertexImagenInlineImage::from_bytes(bytes),
            reference_type: None,
            mask_image_config: None,
        }
    }

    /// Create a reference image from bytes with a reference id.
    pub fn from_bytes_with_id(reference_id: u32, bytes: impl AsRef<[u8]>) -> Self {
        Self {
            reference_id: Some(reference_id),
            reference_image: VertexImagenInlineImage::from_bytes(bytes),
            reference_type: None,
            mask_image_config: None,
        }
    }

    /// Create a Vercel-style raw reference image for editing.
    pub fn raw(reference_id: u32, bytes: impl AsRef<[u8]>) -> Self {
        Self::from_bytes_with_id(reference_id, bytes).with_reference_type("REFERENCE_TYPE_RAW")
    }

    /// Create a Vercel-style mask reference image for editing.
    pub fn mask(reference_id: u32, bytes: impl AsRef<[u8]>) -> Self {
        Self::from_bytes_with_id(reference_id, bytes)
            .with_reference_type("REFERENCE_TYPE_MASK")
            .with_mask_image_config(VertexImagenMaskImageConfig::user_provided())
    }

    /// Set reference type hint.
    pub fn with_reference_type(mut self, reference_type: impl Into<String>) -> Self {
        self.reference_type = Some(reference_type.into());
        self
    }

    /// Set mask image config.
    pub fn with_mask_image_config(mut self, cfg: VertexImagenMaskImageConfig) -> Self {
        self.mask_image_config = Some(cfg);
        self
    }
}

/// Edit options for Vertex Imagen.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VertexImagenEditOptions {
    /// Edit mode (e.g., `EDIT_MODE_INPAINT_INSERTION`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "mode")]
    pub mode: Option<String>,
    /// Base steps (maps to `parameters.editConfig.baseSteps`).
    #[serde(skip_serializing_if = "Option::is_none", rename = "baseSteps")]
    pub base_steps: Option<i64>,
    /// Mask mode (applies when a mask is provided).
    #[serde(skip_serializing_if = "Option::is_none", rename = "maskMode")]
    pub mask_mode: Option<String>,
    /// Mask dilation (applies when a mask is provided).
    #[serde(skip_serializing_if = "Option::is_none", rename = "maskDilation")]
    pub mask_dilation: Option<f64>,
}

impl VertexImagenEditOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = Some(mode.into());
        self
    }

    pub fn with_base_steps(mut self, base_steps: i64) -> Self {
        self.base_steps = Some(base_steps);
        self
    }

    pub fn with_mask_mode(mut self, mask_mode: impl Into<String>) -> Self {
        self.mask_mode = Some(mask_mode.into());
        self
    }

    pub fn with_mask_dilation(mut self, mask_dilation: f64) -> Self {
        self.mask_dilation = Some(mask_dilation);
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
    /// Edit options (used by image editing requests).
    #[serde(skip_serializing_if = "Option::is_none", rename = "edit")]
    pub edit: Option<VertexImagenEditOptions>,
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

    /// Set edit options.
    pub fn with_edit(mut self, edit: VertexImagenEditOptions) -> Self {
        self.edit = Some(edit);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_edit_options_matches_vercel_shape() {
        let options = VertexImagenOptions::new().with_edit(
            VertexImagenEditOptions::new()
                .with_mode("EDIT_MODE_INPAINT_REMOVAL")
                .with_base_steps(50)
                .with_mask_mode("MASK_MODE_USER_PROVIDED")
                .with_mask_dilation(0.01),
        );

        let v = serde_json::to_value(options).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
              "edit": {
                "mode": "EDIT_MODE_INPAINT_REMOVAL",
                "baseSteps": 50,
                "maskMode": "MASK_MODE_USER_PROVIDED",
                "maskDilation": 0.01
              }
            })
        );
    }

    #[test]
    fn serde_reference_images_supports_edit_mask_shape() {
        let options = VertexImagenOptions::new().with_reference_images(vec![
            VertexImagenReferenceImage::raw(1, b"hello"),
            VertexImagenReferenceImage::mask(2, b"world").with_mask_image_config(
                VertexImagenMaskImageConfig::user_provided().with_dilation(0.01),
            ),
        ]);

        let v = serde_json::to_value(options).unwrap();
        assert_eq!(
            v,
            serde_json::json!({
              "referenceImages": [
                {
                  "referenceId": 1,
                  "referenceImage": { "bytesBase64Encoded": "aGVsbG8=" },
                  "referenceType": "REFERENCE_TYPE_RAW"
                },
                {
                  "referenceId": 2,
                  "referenceImage": { "bytesBase64Encoded": "d29ybGQ=" },
                  "referenceType": "REFERENCE_TYPE_MASK",
                  "maskImageConfig": {
                    "maskMode": "MASK_MODE_USER_PROVIDED",
                    "dilation": 0.01
                  }
                }
              ]
            })
        );
    }
}
