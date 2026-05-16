//! Shared stream-route profile helpers.

use crate::BridgeTarget;

#[derive(Debug, Clone, Copy)]
pub(super) struct StreamBridgeProfile {
    pub source: Option<BridgeTarget>,
    pub target: BridgeTarget,
    pub cross_protocol_lossy: bool,
    #[cfg(feature = "openai")]
    pub requires_openai_responses_stream_adapter: bool,
    pub path_label: &'static str,
}

pub(super) fn stream_bridge_profile(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
) -> StreamBridgeProfile {
    let cross_protocol_lossy = matches!(source, Some(source) if source != target);
    #[cfg(feature = "openai")]
    let requires_openai_responses_stream_adapter = matches!(target, BridgeTarget::OpenAiResponses)
        && !matches!(source, Some(BridgeTarget::OpenAiResponses));
    #[cfg(not(feature = "openai"))]
    let requires_openai_responses_stream_adapter = false;
    let path_label = if requires_openai_responses_stream_adapter {
        "openai-responses-stream-adapter"
    } else {
        "protocol-event-serialization"
    };

    StreamBridgeProfile {
        source,
        target,
        cross_protocol_lossy,
        #[cfg(feature = "openai")]
        requires_openai_responses_stream_adapter,
        path_label,
    }
}
