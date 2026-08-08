//! Dated Volcengine ARK model advisories.
//!
//! These constants are ergonomic hints, not an allowlist. Any future model identifier remains
//! callable through the protocol-baseline Chat Completions and Responses paths.

/// Current high-capability Doubao Seed 2.1 Pro snapshot.
pub const DOUBAO_SEED_2_1_PRO_260628: &str = "doubao-seed-2-1-pro-260628";

/// Current low-latency Doubao Seed 2.1 Turbo snapshot.
pub const DOUBAO_SEED_2_1_TURBO_260628: &str = "doubao-seed-2-1-turbo-260628";

/// Doubao Seed 2.0 Pro snapshot.
pub const DOUBAO_SEED_2_0_PRO_260215: &str = "doubao-seed-2-0-pro-260215";

/// Doubao Seed 2.0 Lite snapshot.
pub const DOUBAO_SEED_2_0_LITE_260428: &str = "doubao-seed-2-0-lite-260428";

/// Doubao Seed 2.0 Mini snapshot.
pub const DOUBAO_SEED_2_0_MINI_260428: &str = "doubao-seed-2-0-mini-260428";

/// Doubao Seed 2.0 Code preview snapshot.
pub const DOUBAO_SEED_2_0_CODE_PREVIEW_260215: &str = "doubao-seed-2-0-code-preview-260215";

/// Rolling Doubao Seed Evolving alias.
pub const DOUBAO_SEED_EVOLVING: &str = "doubao-seed-evolving";

/// Recommended general-purpose model hint for this verification snapshot.
pub const DEFAULT_MODEL: &str = DOUBAO_SEED_2_1_PRO_260628;
