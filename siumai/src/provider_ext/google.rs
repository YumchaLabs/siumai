/// Create the Google provider builder.
pub fn google() -> super::gemini::GeminiBuilder {
    crate::compat::Provider::google()
}

/// Create the Google provider builder.
///
/// This is the Rust package-surface analogue of AI SDK `createGoogle()`.
pub fn create_google() -> super::gemini::GeminiBuilder {
    google()
}

/// Create the Google provider builder.
///
/// Deprecated analogue of AI SDK `createGoogleGenerativeAI()`.
#[allow(deprecated)]
#[deprecated(note = "Use `create_google` instead.")]
pub fn create_google_generative_ai() -> super::gemini::GeminiBuilder {
    create_google()
}

pub use super::gemini::*;
