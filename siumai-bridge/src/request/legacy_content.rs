//! Request-side adapters into the legacy `ContentPart` compatibility carrier.
//!
//! Request normalization still has to emit `ChatMessage` / `ContentPart` for stable bridge
//! compatibility. These helpers are the request-side boundary: they may carry request
//! `provider_options`, but they deliberately keep response `provider_metadata` empty.

use siumai_core::types::chat::{FilePartSource, ImageDetail, MediaSource};
use siumai_core::types::{ContentPart, ProviderOptionsMap, ToolResultOutput};

pub(super) fn request_text_part(
    text: impl Into<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Text {
        text: text.into(),
        provider_options,
        provider_metadata: None,
    }
}

pub(super) fn request_reasoning_part(
    text: impl Into<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Reasoning {
        text: text.into(),
        provider_options,
        provider_metadata: None,
    }
}

pub(super) fn request_image_part(
    source: FilePartSource,
    media_type: Option<String>,
    detail: Option<ImageDetail>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Image {
        source,
        media_type,
        detail,
        provider_options,
        provider_metadata: None,
    }
}

pub(super) fn request_audio_part(
    source: MediaSource,
    media_type: Option<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::Audio {
        source,
        media_type,
        provider_options,
        provider_metadata: None,
    }
}

pub(super) fn request_file_part(
    source: FilePartSource,
    media_type: impl Into<String>,
    filename: Option<String>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::File {
        source,
        media_type: media_type.into(),
        filename,
        provider_options,
        provider_metadata: None,
    }
}

pub(super) fn request_tool_call_part(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    arguments: serde_json::Value,
    provider_executed: Option<bool>,
    dynamic: Option<bool>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::ToolCall {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        arguments,
        provider_executed,
        dynamic,
        invalid: None,
        error: None,
        title: None,
        provider_options,
        provider_metadata: None,
    }
}

pub(super) fn request_tool_result_part(
    tool_call_id: impl Into<String>,
    tool_name: impl Into<String>,
    output: ToolResultOutput,
    provider_executed: Option<bool>,
    dynamic: Option<bool>,
    provider_options: ProviderOptionsMap,
) -> ContentPart {
    ContentPart::ToolResult {
        tool_call_id: tool_call_id.into(),
        tool_name: tool_name.into(),
        output,
        input: None,
        provider_executed,
        dynamic,
        preliminary: None,
        title: None,
        provider_options,
        provider_metadata: None,
    }
}
