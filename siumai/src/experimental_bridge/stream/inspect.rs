//! Stream bridge inspection.

use siumai_core::bridge::{BridgeReport, BridgeTarget};

use super::profile::stream_bridge_profile;

/// Inspect a stream bridge route before the stream is consumed.
pub fn inspect_chat_stream_bridge(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    let profile = stream_bridge_profile(source, target);

    if profile.cross_protocol_lossy {
        report.record_lossy_field(
            "stream.protocol",
            format!(
                "stream bridge {} -> {} runs through best-effort event normalization",
                profile.source.expect("profile source").as_str(),
                profile.target.as_str()
            ),
        );
    }
}
