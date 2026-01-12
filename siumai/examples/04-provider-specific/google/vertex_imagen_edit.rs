//! Vertex AI Imagen - edit/inpaint with mask + reference images.
//!
//! Run:
//! ```bash
//! cargo run --example vertex_imagen_edit --features "google-vertex gcp"
//! ```
//!
//! Environment:
//! - `GOOGLE_CLOUD_PROJECT` (or `GCP_PROJECT`)
//! - `GOOGLE_CLOUD_LOCATION` (or `GCP_LOCATION`, default: us-central1)
//! - `INPUT_IMAGE_PATH` (required)
//! - `MASK_IMAGE_PATH` (optional)
//! - `REFERENCE_IMAGE_PATH` (optional)

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::prelude::*;

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::prelude::extensions::ImageExtras;

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
use siumai::provider_ext::google_vertex::options::{
    VertexImagenEditOptions, VertexImagenOptions, VertexImagenReferenceImage,
    VertexImagenRequestExt,
};

#[cfg(all(feature = "google-vertex", feature = "gcp"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let project = std::env::var("GOOGLE_CLOUD_PROJECT")
        .or_else(|_| std::env::var("GCP_PROJECT"))
        .expect("Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT)");
    let location = std::env::var("GOOGLE_CLOUD_LOCATION")
        .or_else(|_| std::env::var("GCP_LOCATION"))
        .unwrap_or_else(|_| "us-central1".to_string());

    // Note: Imagen is accessed via Vertex AI and requires Bearer auth.
    let client = Siumai::builder()
        .google_vertex()
        .base_url_for_vertex(&project, &location, "google")
        .with_gemini_adc()
        // Choose an Imagen edit model as the default.
        .model("imagen-3.0-edit-001")
        .build()
        .await?;

    let input_path = std::env::var("INPUT_IMAGE_PATH").expect("Set INPUT_IMAGE_PATH");
    let input_image = std::fs::read(input_path)?;
    let mask_image = std::env::var("MASK_IMAGE_PATH")
        .ok()
        .map(std::fs::read)
        .transpose()?;
    let reference_image = std::env::var("REFERENCE_IMAGE_PATH")
        .ok()
        .map(std::fs::read)
        .transpose()?;

    let mut options = VertexImagenOptions::new().with_negative_prompt("low quality, blurry");
    if let Some(bytes) = reference_image {
        options = options.with_reference_images(vec![
            VertexImagenReferenceImage::from_bytes(bytes).with_reference_type("SUBJECT"),
        ]);
    }
    if mask_image.is_some() {
        options = options.with_edit(
            VertexImagenEditOptions::new()
                .with_mode("EDIT_MODE_INPAINT_INSERTION")
                .with_mask_mode("MASK_MODE_USER_PROVIDED")
                .with_mask_dilation(0.01),
        );
    }

    let req = siumai::extensions::types::ImageEditRequest {
        image: input_image,
        mask: mask_image,
        prompt: "Fill the masked area with a cute cat".to_string(),
        // Model is required by Vertex Imagen because it is part of the URL.
        model: Some("imagen-3.0-edit-001".to_string()),
        count: Some(1),
        size: Some("1024x1024".to_string()),
        response_format: Some("b64_json".to_string()),
        extra_params: Default::default(),
        provider_options_map: Default::default(),
        http_config: None,
    }
    .with_vertex_imagen_options(options);

    let resp = client.edit_image(req).await?;
    println!("Generated {} image(s)", resp.images.len());
    Ok(())
}

#[cfg(not(all(feature = "google-vertex", feature = "gcp")))]
fn main() {
    eprintln!(
        "This example requires features: google-vertex + gcp.\\n\
Run: cargo run --example vertex_imagen_edit --features \"google-vertex gcp\""
    );
}
