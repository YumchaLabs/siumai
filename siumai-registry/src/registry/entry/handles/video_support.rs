use crate::types::VideoGenerationRequest;

pub(in crate::registry::entry) fn apply_video_handle_default_model(
    mut request: VideoGenerationRequest,
    model_id: &str,
) -> VideoGenerationRequest {
    if request.model.trim().is_empty() && !model_id.trim().is_empty() {
        request.model = model_id.to_string();
    }
    request
}

pub(in crate::registry::entry) fn video_model_handle_max_videos_per_call(
    provider_id: &str,
    _model_id: &str,
) -> Option<u32> {
    match provider_id {
        "gemini" | "google" | "vertex" | "google-vertex" => Some(4),
        "xai" | "minimaxi" => Some(1),
        _ => None,
    }
}
