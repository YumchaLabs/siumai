//! Shared wrapper macros for experimental bridge convenience functions.

macro_rules! define_request_bridge_wrappers {
    ($feature:literal, $plain_fn:ident, $options_fn:ident, $target:expr, $label:literal) => {
        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "`.")]
        pub fn $plain_fn(
            request: &ChatRequest,
            source: Option<BridgeTarget>,
            mode: BridgeMode,
        ) -> Result<BridgeResult<serde_json::Value>, LlmError> {
            bridge_chat_request_to_json(request, source, $target, mode)
        }

        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "` with bridge customization.")]
        pub fn $options_fn(
            request: &ChatRequest,
            source: Option<BridgeTarget>,
            options: BridgeOptions,
        ) -> Result<BridgeResult<serde_json::Value>, LlmError> {
            bridge_chat_request_to_json_with_options(request, source, $target, options)
        }
    };
}

macro_rules! define_response_bridge_wrappers {
    (
        $feature:literal,
        $bytes_fn:ident,
        $bytes_options_fn:ident,
        $value_fn:ident,
        $value_options_fn:ident,
        $target:expr,
        $label:literal
    ) => {
        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "`.")]
        pub fn $bytes_fn(
            response: &ChatResponse,
            source: Option<BridgeTarget>,
            mode: BridgeMode,
            opts: JsonEncodeOptions,
        ) -> Result<BridgeResult<Vec<u8>>, LlmError> {
            bridge_chat_response_to_json_bytes(response, source, $target, mode, opts)
        }

        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "` with bridge customization.")]
        pub fn $bytes_options_fn(
            response: &ChatResponse,
            source: Option<BridgeTarget>,
            options: BridgeOptions,
            opts: JsonEncodeOptions,
        ) -> Result<BridgeResult<Vec<u8>>, LlmError> {
            bridge_chat_response_to_json_bytes_with_options(
                response, source, $target, options, opts,
            )
        }

        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "`.")]
        pub fn $value_fn(
            response: &ChatResponse,
            source: Option<BridgeTarget>,
            mode: BridgeMode,
            opts: JsonEncodeOptions,
        ) -> Result<BridgeResult<serde_json::Value>, LlmError> {
            bridge_chat_response_to_json_value(response, source, $target, mode, opts)
        }

        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "` with bridge customization.")]
        pub fn $value_options_fn(
            response: &ChatResponse,
            source: Option<BridgeTarget>,
            options: BridgeOptions,
            opts: JsonEncodeOptions,
        ) -> Result<BridgeResult<serde_json::Value>, LlmError> {
            bridge_chat_response_to_json_value_with_options(
                response, source, $target, options, opts,
            )
        }
    };
}

macro_rules! define_stream_bridge_wrappers {
    ($feature:literal, $plain_fn:ident, $options_fn:ident, $target:expr, $label:literal) => {
        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "`.")]
        pub fn $plain_fn<S>(
            stream: S,
            source: Option<BridgeTarget>,
            mode: BridgeMode,
        ) -> Result<BridgeResult<ChatByteStream>, LlmError>
        where
            S: futures_util::Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
        {
            bridge_chat_stream_to_bytes(stream, source, $target, mode)
        }

        #[cfg(feature = $feature)]
        #[doc = concat!("Convenience wrapper for `", $label, "` with bridge customization.")]
        pub fn $options_fn<S>(
            stream: S,
            source: Option<BridgeTarget>,
            options: BridgeOptions,
        ) -> Result<BridgeResult<ChatByteStream>, LlmError>
        where
            S: futures_util::Stream<Item = Result<ChatStreamEvent, LlmError>> + Send + 'static,
        {
            bridge_chat_stream_to_bytes_with_options(stream, source, $target, options)
        }
    };
}

pub(crate) use define_request_bridge_wrappers;
pub(crate) use define_response_bridge_wrappers;
pub(crate) use define_stream_bridge_wrappers;
