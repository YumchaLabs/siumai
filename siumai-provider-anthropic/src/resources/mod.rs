//! Provider-native Anthropic resource APIs.
//!
//! These operations have lifecycles distinct from language generation and are
//! intentionally not projected into provider-neutral model traits.

mod common;
mod files;
mod message_batches;
mod skills;
mod tokens;

use std::fmt;
use std::sync::Arc;

use siumai_protocol_anthropic::messages::MessagesAnnotationResolver;
use siumai_transport::ProviderTransport;

pub use files::{
    AnthropicFile, AnthropicFileDeleteResult, AnthropicFileList, AnthropicFileListQuery,
    AnthropicFileUpload, AnthropicFiles,
};
pub use message_batches::{
    AnthropicBatchDeleteResult, AnthropicBatchItem, AnthropicBatchList, AnthropicBatchListQuery,
    AnthropicBatchRequest, AnthropicMessageBatch, AnthropicMessageBatches,
};
pub use skills::{
    AnthropicSkill, AnthropicSkillFile, AnthropicSkillUpload, AnthropicSkillUploadResult,
    AnthropicSkillVersion, AnthropicSkillVersionList, AnthropicSkills,
};
pub use tokens::{AnthropicTokenCount, AnthropicTokens};

pub(crate) struct NativeRuntime {
    pub(crate) transport: ProviderTransport,
    pub(crate) api_version: Arc<str>,
    pub(crate) beta_features: Arc<[String]>,
    pub(crate) annotation_resolver: Arc<dyn MessagesAnnotationResolver>,
}

impl fmt::Debug for NativeRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeRuntime")
            .field("transport", &"shared")
            .field("api_version", &self.api_version)
            .field("beta_features", &self.beta_features)
            .field("annotation_resolver", &"configured")
            .finish()
    }
}
