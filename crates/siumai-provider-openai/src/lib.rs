//! Siumai OpenAI Provider (extracted)
//!
//! This crate hosts the OpenAI provider implementation that is used by the
//! aggregator crate (`siumai`) via feature gates. The goal is to keep
//! provider-specific behavior (headers, routing, standard selection) close to
//! the provider itself while exposing a thin, core-only interface.

pub const VERSION: &str = "0.0.1";

/// Marker type for the OpenAI provider crate. This can be used in tests or
/// debug logs to assert that the external provider wiring is active.
#[derive(Debug, Clone, Default)]
pub struct OpenAiProviderMarker;

/// Lightweight constants and adapter/spec placeholders to mirror aggregator structure.
pub mod constants {
    /// Default OpenAI REST endpoint base.
    pub const OPENAI_V1_ENDPOINT: &str = "https://api.openai.com/v1";
}

pub mod adapter;
pub mod helpers;

pub mod spec {
    //! Core-level OpenAI provider spec implementation.
    //!
    //! This module exposes a `CoreProviderSpec` implementation that can be
    //! consumed by aggregator crates without introducing a dependency from
    //! this crate back to the `siumai` crate. It operates purely on
    //! `siumai-core` types.

    use siumai_core::error::LlmError;
    use siumai_core::execution::chat::ChatInput;
    use siumai_core::execution::streaming::ChatStreamEventCore;
    use siumai_core::provider_spec::{CoreChatTransformers, CoreProviderContext, CoreProviderSpec};
    use siumai_core::traits::ProviderCapabilities;
    use siumai_std_openai::openai::chat::OpenAiChatStandard;

    /// Core OpenAI provider spec backed by the OpenAI standard.
    ///
    /// All OpenAI-specific behavior that is shared across languages lives here:
    /// - Capability declaration
    /// - Header construction
    /// - Routing between Chat Completions and Responses API
    /// - Selection of request/response/stream transformers
    #[derive(Clone, Default)]
    pub struct OpenAiCoreSpec {
        chat_standard: OpenAiChatStandard,
    }

    impl OpenAiCoreSpec {
        /// Create a new core spec using the default OpenAI Chat standard.
        pub fn new() -> Self {
            Self {
                chat_standard: OpenAiChatStandard::new(),
            }
        }

        /// Determine whether to use the Responses API based on core-level extras.
        ///
        /// Aggregator crates are expected to populate `openai.responses_api`
        /// inside `CoreProviderContext::extras` when the user opts in via
        /// provider options.
        fn use_responses_api(&self, ctx: &CoreProviderContext) -> bool {
            let enabled = ctx
                .extras
                .get("openai.responses_api")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            crate::helpers::use_responses_api_from_flag(enabled)
        }
    }

    impl CoreProviderSpec for OpenAiCoreSpec {
        fn id(&self) -> &'static str {
            "openai"
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::new()
                .with_chat()
                .with_streaming()
                .with_tools()
                .with_embedding()
        }

        fn build_headers(
            &self,
            ctx: &CoreProviderContext,
        ) -> Result<reqwest::header::HeaderMap, LlmError> {
            let api_key = ctx
                .api_key
                .as_ref()
                .ok_or_else(|| LlmError::MissingApiKey("OpenAI API key not provided".into()))?;

            crate::helpers::build_openai_json_headers(
                api_key,
                ctx.organization.as_deref(),
                ctx.project.as_deref(),
                &ctx.http_extra_headers,
            )
        }

        fn chat_url(&self, ctx: &CoreProviderContext) -> String {
            // Decide whether to route to Chat Completions or Responses API.
            let use_responses = self.use_responses_api(ctx);
            let suffix = crate::helpers::chat_path(use_responses);
            format!("{}{}", ctx.base_url.trim_end_matches('/'), suffix)
        }

        fn choose_chat_transformers(
            &self,
            _input: &ChatInput,
            ctx: &CoreProviderContext,
        ) -> CoreChatTransformers {
            let req = self
                .chat_standard
                .create_request_transformer(&ctx.provider_id);
            let resp = self
                .chat_standard
                .create_response_transformer(&ctx.provider_id);
            let stream = self.chat_standard.create_stream_converter(&ctx.provider_id);

            CoreChatTransformers {
                request: req,
                response: resp,
                stream: Some(stream),
            }
        }

        fn map_core_stream_event(&self, event: ChatStreamEventCore) -> ChatStreamEventCore {
            event
        }
    }
}
