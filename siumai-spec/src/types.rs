//! Core Data Type Definitions
//!
//! This module contains all data structures used in the LLM library, organized by
//! functionality. The public API is surfaced from this root module; internal
//! submodules like `params`, `http`, and `usage` are implementation details.
//!
//! ## Module Organization
//!
//! - **`chat/`** - Chat-related types (messages, requests, responses, content)
//! - **`common`** - Common enums/metadata shared across providers
//! - **`params`** - Common AI parameters (model, temperature, max_tokens, etc.)
//! - **`http`** - HTTP configuration (`HttpConfig` and builder)
//! - **`usage`** - Token usage and detailed usage breakdown
//! - **`embedding`** - Embedding request/response types
//! - **`image`** - Image generation types
//! - **`audio`** - Audio transcription/generation types
//! - **`tools`** - Tool/function calling types
//! - **`streaming`** - Streaming response types
//! - **`provider_options/`** - Provider options transport helpers (provider-agnostic)
//! - **`provider_metadata/`** - Provider-specific response metadata
//! - **`provider_options_map/`** - Open provider options map (provider-id keyed JSON object)
//!
//! ## Usage Guidelines
//!
//! ### For Application Developers
//!
//! Most types are re-exported at the module root for convenience:
//!
//! ```rust
//! use siumai::types::{ChatMessage, ChatRequest, ChatResponse, CommonParams};
//! ```
//!
//! ### For Library Developers
//!
//! When adding new types:
//! - **Common enums/metadata** → `common.rs`
//! - **Shared AI parameters** → `params.rs`
//! - **HTTP configuration** → `http.rs`
//! - **Usage accounting** → `usage.rs`
//! - **Chat-related** → `chat/` subdirectory
//! - **Provider-specific typed options/metadata** → provider crates; this crate only carries the
//!   provider-id keyed JSON maps used to transport that data.
//!
//! ## Type Categories
//!
//! ### Request Types
//! - `ChatRequest` - Chat completion requests
//! - `EmbeddingRequest` - Embedding generation requests
//! - `ImageGenerationRequest` - Image generation requests
//! - `AudioRequest` - Audio transcription/generation requests
//!
//! ### Response Types
//! - `ChatResponse` - Chat completion responses
//! - `EmbeddingResponse` - Embedding vectors
//! - `ImageGenerationResponse` - Generated images
//! - `AudioResponse` - Audio transcription/generation results
//!
//! ### Common Types
//! - `CommonParams` - Parameters shared across all providers
//! - `HttpConfig` - HTTP configuration shared across providers
//! - `Usage` - Token usage information
//! - `FinishReason` - Completion finish reasons
//! - `ProviderType` - Legacy compatibility provider classification enum
//! - `ResponseMetadata` - Shared response metadata
//!
//! ### Provider-Specific Types
//! Provider-specific typed options/metadata are intentionally **not** owned by `siumai-core`.
//! They live in provider crates to reduce coupling and compile cost.

pub mod ai_sdk;
pub mod audio;
pub mod chat;
pub mod common;
pub mod completion;
pub mod embedding;
pub mod files;
pub mod http;
pub mod image;
pub mod models;
pub mod moderation;
pub mod music;
pub mod params;
pub mod prompt;
pub mod provider_metadata;
pub mod provider_options;
pub mod provider_options_map;
pub mod rerank;
pub mod schema;
pub mod skills;
pub mod stream_options;
pub mod streaming;
pub mod tools;
pub mod usage;
pub mod video;

/// Explicit compatibility namespace for legacy spec-level carriers.
///
/// This namespace is for migration and serde-facing compatibility payloads that intentionally keep
/// older broad shapes. New request code should use prompt/model-message parts, and new response code
/// should use generated-output parts.
pub mod compat {
    /// Legacy chat content carriers.
    pub mod content {
        pub use super::super::chat::compat::*;
    }
}

/// Directional content namespaces.
///
/// These modules make the preferred content direction visible without removing the historical root
/// exports during the current compatibility window:
///
/// - `content::prompt` is request/input oriented and carries prompt provider options.
/// - `content::output` is response/generated-output oriented and carries provider metadata.
/// - `content::compat` is the explicit namespace for legacy serde-facing chat payloads.
pub mod content {
    /// Request-side prompt and model-message content.
    pub mod prompt {
        pub use super::super::prompt::{
            AssistantContent, AssistantContentPart, AssistantModelMessage, CustomPart, DataContent,
            FilePart, ImagePart, InvalidDataContentError, MissingToolResultsError, ModelMessage,
            ModelMessageConversionError, ModelMessageRole, Prompt, PromptExecutionError,
            PromptInput, PromptValidationError, ReasoningFilePart, ReasoningPart,
            StandardizedPrompt, SystemModelMessage, SystemPrompt, TextPart, ToolApprovalRequest,
            ToolApprovalResponse, ToolCallPart, ToolContent, ToolContentPart, ToolModelMessage,
            ToolResultPart, UserContent, UserContentPart, UserModelMessage,
            convert_data_content_to_base64_string, convert_data_content_to_uint8_array,
            convert_uint8_array_to_text, project_chat_message_to_prompt_message,
            project_chat_messages_to_prompt_messages, project_prompt_message_to_chat_message,
            project_prompt_messages_to_chat_messages,
        };
    }

    /// Response-side generated-output content and lossless response projection helpers.
    pub mod output {
        pub use super::super::ai_sdk::{
            CustomOutput, DefaultGeneratedFile, DefaultGeneratedFileWithType, DynamicToolCall,
            DynamicToolError, DynamicToolResult, FileOutput, GenerateTextContentPart,
            GenerateTextContentPartProjectionError, GenerateTextReasoningPart,
            GenerateTextStepReasoningPart, GeneratedFile, ReasoningFileOutput, ReasoningOutput,
            ResponseMessage, Source, StaticToolCall, StaticToolError, StaticToolOutputDenied,
            StaticToolResult, TextOutput, ToolApprovalRequestOutput, ToolApprovalResponseOutput,
            ToolCall, ToolError, ToolOutput, ToolOutputDenied, ToolResult, TypedToolCall,
            TypedToolError, TypedToolOutputDenied, TypedToolResult,
            project_chat_response_to_generate_text_content_parts,
            project_response_content_part_to_generate_text_content_part,
            project_response_content_to_generate_text_content_parts,
        };
    }

    /// Legacy chat content carriers.
    pub mod compat {
        pub use super::super::compat::content::*;
    }
}

// Re-export all types for convenience
pub use ai_sdk::*;
pub use audio::*;
pub use chat::*;
pub use common::*;
pub use completion::*;
pub use embedding::*;
pub use files::*;
pub use http::*;
pub use image::*;
pub use models::*;
pub use moderation::*;
pub use music::*;
pub use params::*;
pub use prompt::*;
pub use provider_metadata::*;
pub use provider_options::*;
pub use provider_options_map::*;
pub use rerank::*;
pub use schema::*;
pub use skills::*;
pub use stream_options::*;
pub use streaming::*;
pub use tools::*;
pub use usage::*;
pub use video::*;

// Provider-specific typed metadata types are intentionally owned by provider crates.
