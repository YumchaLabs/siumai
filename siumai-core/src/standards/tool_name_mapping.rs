//! Compatibility re-export for tool name mapping helpers.
//!
//! The implementation moved to `siumai-provider-utils` in FCAB-120. Keep this module as a
//! migration alias while facade and provider/protocol callers use the provider-utils owner.

pub use siumai_provider_utils::standards::{ToolNameMapping, create_tool_name_mapping};
