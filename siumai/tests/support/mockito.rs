//! mockito test utilities
//!
//! Goals:
//! - Provide a unified async Server creation and common JSON response helpers
//! - Wrap `server.url()` and similar helpers to reduce direct API coupling
//! - Insulate tests from future mockito API changes

#![allow(dead_code)]

use mockito::{Matcher, Server, ServerGuard};

/// Start an async mockito Server
pub async fn start() -> ServerGuard {
    Server::new_async().await
}

/// Get the Server base URL (including scheme)
pub fn url(server: &ServerGuard) -> String {
    server.url()
}

/// Convenience regex matcher helper
pub fn regex(re: &str) -> Matcher {
    Matcher::Regex(re.to_string())
}

/// Convenience helper to create a JSON response mock
/// (automatically sets `content-type: application/json`)
pub async fn json_mock<P: Into<Matcher>>(
    server: &mut ServerGuard,
    method: &str,
    path: P,
    status: u16,
    body_json: &str,
) {
    server
        .mock(method, path)
        .with_status(status as usize)
        .with_header("content-type", "application/json")
        .with_body(body_json)
        .create_async()
        .await;
}
