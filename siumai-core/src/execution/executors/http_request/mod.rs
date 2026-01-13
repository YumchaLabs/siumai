//! Basic HTTP request helpers (non-stream)
//!
//! Provides stable entry points for common GET/DELETE/POST JSON/multipart requests,
//! including interceptor hooks and a single 401 retry with rebuilt headers.

pub use crate::execution::executors::common::{
    HttpBinaryResult, HttpBody, HttpExecutionConfig, HttpExecutionResult,
};

mod bytes;
mod json;
mod multipart;
mod streaming_response;
mod verbs;

pub use bytes::{execute_bytes_request, execute_multipart_bytes_request};
#[allow(deprecated)]
pub use json::execute_json_request_with_headers;
pub use json::{execute_json_request, execute_request};
pub use multipart::execute_multipart_request;
pub use streaming_response::{
    execute_json_request_streaming_response, execute_json_request_streaming_response_with_ctx,
    execute_multipart_request_streaming_response,
    execute_multipart_request_streaming_response_with_ctx,
};
pub use verbs::{
    execute_delete_json_request, execute_delete_request, execute_get_binary, execute_get_request,
    execute_patch_json_request,
};

#[cfg(test)]
mod tests;
