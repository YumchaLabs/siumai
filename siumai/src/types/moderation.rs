//! Content moderation types

use std::collections::HashMap;

/// Moderation request
#[derive(Debug, Clone)]
pub struct ModerationRequest {
    /// Input text to moderate (single input). If `inputs` is provided, this
    /// field is ignored when constructing provider JSON (OpenAI Moderations API
    /// supports string or array for the same `input` key).
    pub input: String,
    /// Multiple input texts to moderate. When present and non-empty, the
    /// transformer will encode `input` as an array.
    pub inputs: Option<Vec<String>>,
    /// Model to use for moderation
    pub model: Option<String>,
}

/// Moderation response
#[derive(Debug, Clone)]
pub struct ModerationResponse {
    /// Moderation results
    pub results: Vec<ModerationResult>,
    /// Model used
    pub model: String,
}

/// Individual moderation result
#[derive(Debug, Clone)]
pub struct ModerationResult {
    /// Whether content was flagged
    pub flagged: bool,
    /// Category scores
    pub categories: HashMap<String, bool>,
    /// Category confidence scores
    pub category_scores: HashMap<String, f32>,
}
