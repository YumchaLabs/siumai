//! Directional content facade.
//!
//! Prefer `content::prompt` for request input and `content::output` for generated response output.
//! Legacy serde-facing chat payloads remain explicit under `content::compat` / `compat::content`.

/// Request-side prompt and model-message content.
pub mod prompt {
    pub use siumai_core::types::content::prompt::*;
}

/// Response-side generated-output content and projection helpers.
pub mod output {
    pub use siumai_core::types::content::output::*;
}

/// Legacy chat content carriers for migration and serde compatibility.
pub mod compat {
    pub use crate::compat::content::*;
}
