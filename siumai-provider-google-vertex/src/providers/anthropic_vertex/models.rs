//! Dated ergonomic hints for Claude models exposed through Vertex AI.

/// Claude Opus 5 on Vertex AI.
pub const CLAUDE_OPUS_5: &str = "claude-opus-5";
/// Claude Sonnet 5 on Vertex AI.
pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";
/// Claude Fable 5 on Vertex AI.
pub const CLAUDE_FABLE_5: &str = "claude-fable-5";
/// Pinned Claude Haiku 4.5 release on Vertex AI.
pub const CLAUDE_HAIKU_4_5_20251001: &str = "claude-haiku-4-5@20251001";
/// Claude Opus 4.8 on Vertex AI.
pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";
/// Claude Opus 4.7 on Vertex AI.
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";
/// Claude Opus 4.6 on Vertex AI.
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";
/// Claude Sonnet 4.6 on Vertex AI.
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
/// Pinned Claude Opus 4.5 release on Vertex AI.
pub const CLAUDE_OPUS_4_5_20251101: &str = "claude-opus-4-5@20251101";
/// Pinned Claude Sonnet 4.5 release on Vertex AI.
pub const CLAUDE_SONNET_4_5_20250929: &str = "claude-sonnet-4-5@20250929";

pub(crate) const CLAUDE_OPUS_4_1_20250805: &str = "claude-opus-4-1@20250805";
pub(crate) const CLAUDE_OPUS_4_20250514: &str = "claude-opus-4@20250514";
pub(crate) const CLAUDE_SONNET_4_20250514: &str = "claude-sonnet-4@20250514";

/// Current model hints verified from Google documentation on 2026-08-06.
///
/// The list is advisory. Callers may still select future or pinned model IDs.
pub const fn current_models() -> [&'static str; 10] {
    [
        CLAUDE_OPUS_5,
        CLAUDE_SONNET_5,
        CLAUDE_FABLE_5,
        CLAUDE_OPUS_4_8,
        CLAUDE_OPUS_4_7,
        CLAUDE_SONNET_4_6,
        CLAUDE_OPUS_4_6,
        CLAUDE_OPUS_4_5_20251101,
        CLAUDE_SONNET_4_5_20250929,
        CLAUDE_HAIKU_4_5_20251001,
    ]
}

pub(crate) fn supports_structured_outputs(model: &str) -> bool {
    matches!(
        model,
        CLAUDE_OPUS_5
            | CLAUDE_SONNET_5
            | CLAUDE_FABLE_5
            | CLAUDE_HAIKU_4_5_20251001
            | CLAUDE_OPUS_4_8
            | CLAUDE_OPUS_4_7
            | CLAUDE_OPUS_4_6
            | CLAUDE_SONNET_4_6
            | CLAUDE_OPUS_4_5_20251101
            | CLAUDE_SONNET_4_5_20250929
    )
}

pub(crate) fn supports_one_hour_cache(model: &str) -> bool {
    matches!(
        model,
        CLAUDE_OPUS_5
            | CLAUDE_SONNET_5
            | CLAUDE_FABLE_5
            | CLAUDE_HAIKU_4_5_20251001
            | CLAUDE_OPUS_4_8
            | CLAUDE_OPUS_4_7
            | CLAUDE_OPUS_4_6
            | CLAUDE_SONNET_4_6
            | CLAUDE_OPUS_4_5_20251101
            | CLAUDE_SONNET_4_5_20250929
            | CLAUDE_OPUS_4_1_20250805
            | CLAUDE_OPUS_4_20250514
            | CLAUDE_SONNET_4_20250514
    )
}

pub(crate) fn uses_strict_sampling(model: &str) -> bool {
    matches!(
        model,
        CLAUDE_OPUS_5 | CLAUDE_SONNET_5 | CLAUDE_FABLE_5 | CLAUDE_OPUS_4_8 | CLAUDE_OPUS_4_7
    )
}
