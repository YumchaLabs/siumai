//! Stream bridge inspection.

use siumai_core::bridge::{BridgeReport, BridgeTarget, BridgeWarning, BridgeWarningKind};

/// Inspect a stream bridge route before the stream is consumed.
pub fn inspect_chat_stream_bridge(
    source: Option<BridgeTarget>,
    target: BridgeTarget,
    report: &mut BridgeReport,
) {
    if let Some(source) = source
        && source != target
    {
        report.add_warning(BridgeWarning::new(
            BridgeWarningKind::Custom,
            format!(
                "stream bridge {} -> {} runs through best-effort event normalization",
                source.as_str(),
                target.as_str()
            ),
        ));
    }
}
