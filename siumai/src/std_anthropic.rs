//! Bridge module for Anthropic standard (external-only re-export)

pub mod anthropic {
    pub mod chat {
        pub use siumai_std_anthropic::anthropic::chat::*;
    }
}
