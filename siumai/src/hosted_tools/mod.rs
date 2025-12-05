//! Provider-Defined Tools
//!
//! This module provides factory functions for creating provider-specific tools that are
//! executed by the provider's servers (e.g., web search, file search, code execution).
//!
//! # Overview
//!
//! Provider-defined tools are different from user-defined function tools:
//! - **User-defined functions**: You provide the implementation, the model calls your code
//! - **Provider-defined tools**: The provider executes the tool on their servers
//!
//! # Supported Providers
//!
//! - **OpenAI**: Web search, file search, computer use
//! - **Anthropic**: Web search (2025-03-05 version)
//! - **Google/Gemini**: Code execution
//!
//! # Examples
//!
//! ## OpenAI Web Search
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//! use siumai::hosted_tools::openai;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Siumai::builder()
//!     .openai()
//!     .api_key("your-api-key")
//!     .model("gpt-4")
//!     .build()
//!     .await?;
//!
//! let messages = vec![user!("What's the latest news about Rust?")];
//!
//! // Create a web search tool
//! let web_search = openai::web_search()
//!     .with_search_context_size("high")
//!     .build();
//!
//! let (response, _) = client.chat()
//!     .messages(&messages)
//!     .tools(&[web_search])
//!     .execute()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Anthropic Web Search
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//! use siumai::hosted_tools::anthropic;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Siumai::builder()
//!     .anthropic()
//!     .api_key("your-api-key")
//!     .model("claude-3-5-sonnet-20241022")
//!     .build()
//!     .await?;
//!
//! let messages = vec![user!("Search for Rust async programming tutorials")];
//!
//! // Create a web search tool with domain restrictions
//! let web_search = anthropic::web_search_20250305()
//!     .with_max_uses(3)
//!     .with_allowed_domains(vec!["github.com".to_string(), "docs.rs".to_string()])
//!     .build();
//!
//! let (response, _) = client.chat()
//!     .messages(&messages)
//!     .tools(&[web_search])
//!     .execute()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Google Code Execution
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//! use siumai::hosted_tools::google;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Siumai::builder()
//!     .gemini()
//!     .api_key("your-api-key")
//!     .model("gemini-2.0-flash-exp")
//!     .build()
//!     .await?;
//!
//! let messages = vec![user!("Calculate the first 10 Fibonacci numbers")];
//!
//! // Create a code execution tool
//! let code_exec = google::code_execution();
//!
//! let (response, _) = client.chat()
//!     .messages(&messages)
//!     .tools(&[code_exec])
//!     .execute()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Mixing Hosted Tools with User Functions
//!
//! You can mix provider-defined tools with user-defined function tools:
//!
//! ```rust,ignore
//! use siumai::prelude::*;
//! use siumai::hosted_tools::openai;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Siumai::builder()
//!     .openai()
//!     .api_key("your-api-key")
//!     .model("gpt-4")
//!     .build()
//!     .await?;
//!
//! let messages = vec![user!("Search for weather in SF and get the temperature")];
//!
//! // Provider-hosted tool (executed by OpenAI)
//! let web_search = openai::web_search().build();
//!
//! // User-defined function (executed by your code)
//! let get_weather = tool!("get_weather", "Get current weather", {
//!     "location": String
//! });
//!
//! let (response, _) = client.chat()
//!     .messages(&messages)
//!     .tools(&[web_search, get_weather])
//!     .execute()
//!     .await?;
//! # Ok(())
//! # }
//! ```

pub mod anthropic;
pub mod google;
pub mod openai;
