//! AI SDK-style runtime metadata helpers.

/// Version string for the current Rust package.
///
/// AI SDK provider-utils injects this at package build time. In Rust, Cargo
/// provides the equivalent crate version through `CARGO_PKG_VERSION`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Return a deterministic user-agent suffix for the Rust runtime.
///
/// JavaScript AI SDK detects browser, Node, Deno, Bun, Workers, and Edge from
/// host globals. Rust has no equivalent host-object namespace, so this helper
/// exposes the truthful runtime marker used by Siumai user-agent composition.
pub fn get_runtime_environment_user_agent() -> String {
    "runtime/rust".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_runtime_user_agent_and_version() {
        assert_eq!(get_runtime_environment_user_agent(), "runtime/rust");
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }
}
