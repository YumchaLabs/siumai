//! Provider-native OpenAI resource wire contracts.
//!
//! These resources have lifecycles distinct from model generation. The
//! protocol crate owns their JSON shapes while the provider crate owns HTTP
//! execution, authentication, validation, and replay policy.

mod common;
mod conversations;
mod files;
pub(crate) mod skills;
mod vector_stores;

pub use common::{OpenAiCursorPage, OpenAiListOrder, OpenAiMetadata, OpenAiResourceCodecError};
pub use conversations::{
    OpenAiConversation, OpenAiConversationCreateRequest, OpenAiConversationDeleted,
    OpenAiConversationInputItem, OpenAiConversationItem, OpenAiConversationItemsCreateRequest,
    OpenAiConversationRole, OpenAiConversationUpdateRequest,
};
pub use files::{
    OpenAiFile, OpenAiFileDeleted, OpenAiFileExpirationAnchor, OpenAiFileExpiresAfter,
    OpenAiFilePurpose, OpenAiFileUploadPurpose,
};
pub use vector_stores::{
    OpenAiChunkingStrategy, OpenAiStaticChunkingSettings, OpenAiVectorStore,
    OpenAiVectorStoreCreateRequest, OpenAiVectorStoreDeleted, OpenAiVectorStoreExpiration,
    OpenAiVectorStoreExpirationAnchor, OpenAiVectorStoreFile, OpenAiVectorStoreFileAttachRequest,
    OpenAiVectorStoreFileCounts, OpenAiVectorStoreFileDeleted, OpenAiVectorStoreFileError,
    OpenAiVectorStoreUpdateRequest,
};
