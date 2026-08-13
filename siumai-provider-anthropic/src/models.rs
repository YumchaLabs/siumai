//! Evidence-backed Anthropic model hints.
//!
//! These constants are conveniences, not an allowlist. Anthropic model IDs remain
//! open input and unknown future IDs are passed through with protocol-baseline behavior.

/// Claude Opus 5 pinned model ID, verified on 2026-08-06.
pub const CLAUDE_OPUS_5: &str = "claude-opus-5";
/// Claude Sonnet 5 pinned model ID, verified on 2026-08-06.
pub const CLAUDE_SONNET_5: &str = "claude-sonnet-5";
/// Claude Fable 5 pinned model ID, verified on 2026-08-06.
pub const CLAUDE_FABLE_5: &str = "claude-fable-5";
/// Claude Mythos 5 pinned model ID, verified on 2026-08-06.
pub const CLAUDE_MYTHOS_5: &str = "claude-mythos-5";
/// Claude Mythos Preview model ID, currently deprecated in favor of Mythos 5.
pub const CLAUDE_MYTHOS_PREVIEW: &str = "claude-mythos-preview";
/// Claude Haiku 4.5 rolling model ID, verified on 2026-08-06.
pub const CLAUDE_HAIKU_4_5: &str = "claude-haiku-4-5";
/// Claude Haiku 4.5 pinned snapshot ID, verified on 2026-08-06.
pub const CLAUDE_HAIKU_4_5_20251001: &str = "claude-haiku-4-5-20251001";
/// Claude Opus 4.8 pinned model ID used as the current Opus 4.1 replacement.
pub const CLAUDE_OPUS_4_8: &str = "claude-opus-4-8";
/// Claude Opus 4.7 pinned model ID, verified on 2026-08-06.
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4-7";
/// Claude Opus 4.6 pinned model ID, verified on 2026-08-06.
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4-6";
/// Claude Sonnet 4.6 pinned model ID, verified on 2026-08-06.
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4-6";
/// Claude Opus 4.1 snapshot retired on 2026-08-05.
pub const CLAUDE_OPUS_4_1_20250805: &str = "claude-opus-4-1-20250805";

/// Current recommended model IDs in stable catalog order.
pub const fn current_models() -> [&'static str; 4] {
    [
        CLAUDE_OPUS_5,
        CLAUDE_SONNET_5,
        CLAUDE_FABLE_5,
        CLAUDE_HAIKU_4_5,
    ]
}
