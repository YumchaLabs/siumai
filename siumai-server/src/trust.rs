use std::fmt;

use siumai_core::RouteId;
use siumai_runtime::approval::TrustIdentity;

/// Host-authenticated identity and route binding for a server request.
///
/// This type deliberately implements neither `Deserialize` nor a constructor
/// from HTTP headers. The host application authenticates the request first and
/// only then constructs this context.
#[derive(Clone, PartialEq, Eq)]
pub struct ServerTrustContext {
    identity: TrustIdentity,
    route: RouteId,
}

impl ServerTrustContext {
    pub fn new(identity: TrustIdentity, route: RouteId) -> Self {
        Self { identity, route }
    }

    pub fn identity(&self) -> &TrustIdentity {
        &self.identity
    }

    pub fn route(&self) -> &RouteId {
        &self.route
    }
}

impl fmt::Debug for ServerTrustContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ServerTrustContext")
            .field("route", &self.route)
            .field("identity", &"<redacted>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_authenticated_identity() {
        let identity = TrustIdentity::new("issuer", "audience", "subject", "tenant").unwrap();
        let context = ServerTrustContext::new(identity, RouteId::new("tools").unwrap());
        let debug = format!("{context:?}");

        assert!(debug.contains("tools"));
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("subject"));
        assert!(!debug.contains("tenant"));
    }
}
