//! Provider-Model Architecture
//!
//! This module implements a layered architecture that separates provider management,
//! model configuration, and execution logic.
//!
//! ## Architecture Overview
//!
//! ```text
//! Provider (Factory)
//!     ↓
//! Model (Endpoint Configuration)
//!     ↓
//! Standard (Reusable Implementation) + Adapter (Handle Differences)
//!     ↓
//! Transformer (Format Conversion)
//!     ↓
//! Executor (HTTP Execution + Middleware + Interceptor + Retry)
//! ```
//!
//! ## Key Concepts
//!
//! - **Provider**: Lightweight factory that creates Model instances
//! - **Model**: Encapsulates endpoint-specific configuration and Executor creation
//! - **Standard**: Provides reusable API format implementations (e.g., OpenAI format)
//! - **Adapter**: Handles provider-specific differences
//! - **Transformer**: Handles request/response format conversion
//! - **Executor**: Executes actual HTTP requests with middleware/interceptor/retry support
//!
//! ## Example
//!
//! ```rust,ignore
//! // Create provider
//! let provider = OpenAiProvider::new(config);
//!
//! // Get chat model
//! let model = provider.chat("gpt-4")?;
//!
//! // Create executor with middleware/interceptor/retry
//! let executor = model.create_executor(
//!     http_client,
//!     vec![],  // interceptors
//!     vec![],  // middlewares
//!     None,    // retry_options
//! );
//!
//! // Execute request
//! let response = executor.execute(request).await?;
//! ```

pub mod model;
pub mod provider;

pub use model::{ChatModel, EmbeddingModel, ImageModel, RerankModel};
pub use provider::Provider;
