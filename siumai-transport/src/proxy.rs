//! Explicit trusted HTTPS CONNECT routing for provider HTTP.

use std::fmt;
use std::net::SocketAddr;

use url::Url;

use crate::{
    CredentialAudience, EndpointConfig, EndpointError, EndpointPolicy, LocalNetworkGrant,
    TransportConfigError,
};

const MAX_PROXY_CREDENTIAL_BYTES: usize = 256;

/// A separately validated origin used to reach a forward proxy.
///
/// Proxy endpoints are deliberately distinct from provider endpoints. The
/// origin must be rooted at `/`, must not carry query state, and may use either
/// a public HTTPS policy or an explicitly granted local HTTP(S) policy.
#[derive(Clone)]
pub struct ProxyEndpoint {
    endpoint: EndpointConfig,
}

impl ProxyEndpoint {
    /// Construct a proxy endpoint from an explicit endpoint policy.
    pub fn new(origin: impl AsRef<str>, policy: EndpointPolicy) -> Result<Self, EndpointError> {
        if matches!(&policy, EndpointPolicy::Official(_)) {
            return Err(EndpointError::ProxyPolicyNotAllowed);
        }
        let endpoint = EndpointConfig::new(origin, policy)?;
        let url = endpoint.expose_base_url();
        if url.path() != "/" || url.query().is_some() {
            return Err(EndpointError::ProxyOriginMustBeRoot);
        }
        Ok(Self { endpoint })
    }

    /// Construct a public HTTPS proxy origin.
    pub fn https(origin: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(origin, EndpointPolicy::PublicCustom)
    }

    /// Construct a loopback proxy origin with an explicit local grant.
    pub fn local_explicit(origin: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            origin,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
        )
    }

    /// Construct a private-network proxy origin with an explicit local grant.
    pub fn private_network_explicit(origin: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            origin,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::PrivateNetwork),
        )
    }

    /// Construct a link-local proxy origin with an explicit local grant.
    pub fn link_local_explicit(origin: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            origin,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::LinkLocal),
        )
    }

    /// Construct an RFC 6598 shared-address-space proxy origin.
    pub fn shared_address_space_explicit(origin: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            origin,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::SharedAddressSpace),
        )
    }

    /// Return the proxy endpoint policy.
    pub fn policy(&self) -> EndpointPolicy {
        self.endpoint.policy()
    }

    /// Return the exact audience used to authenticate the proxy connection.
    pub fn audience(&self) -> &CredentialAudience {
        self.endpoint.audience()
    }

    /// Return whether the proxy connection itself uses TLS.
    pub fn uses_tls(&self) -> bool {
        self.endpoint.expose_base_url().scheme() == "https"
    }

    /// Resolve the proxy connector host under this endpoint's network policy.
    ///
    /// This low-level operation exists so sibling infrastructure crates can
    /// reuse the route's DNS authority without receiving a raw HTTP client.
    #[doc(hidden)]
    pub async fn resolve_for_connector(
        &self,
        requested_host: &str,
        resolver: &dyn crate::Resolver,
    ) -> Result<Vec<SocketAddr>, EndpointError> {
        self.endpoint
            .resolve_for_connector(requested_host, resolver)
            .await
    }

    /// Validate the connected proxy peer under this endpoint's network policy.
    #[doc(hidden)]
    pub fn validate_remote(&self, remote: SocketAddr) -> Result<(), EndpointError> {
        self.endpoint.validate_remote(remote)
    }

    pub(crate) fn endpoint(&self) -> &EndpointConfig {
        &self.endpoint
    }

    pub(crate) fn url(&self) -> &Url {
        self.endpoint.expose_base_url()
    }
}

impl fmt::Debug for ProxyEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProxyEndpoint")
            .field("policy", &self.endpoint.policy())
            .field("audience", &self.endpoint.audience())
            .field("origin", &"[REDACTED]")
            .finish()
    }
}

/// An immutable, bounded Basic credential used only during proxy negotiation.
#[derive(Clone)]
pub struct ProxyBasicCredential {
    username: String,
    password: String,
}

impl ProxyBasicCredential {
    /// Construct a header-safe proxy Basic credential.
    ///
    /// The value is an immutable snapshot. Rotate credentials by rebuilding the
    /// configured provider transport; no refresh callback or global store is
    /// part of the transport contract.
    pub fn new(
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Result<Self, TransportConfigError> {
        let username = username.into();
        validate_credential_field("proxy_username", &username)?;
        if username.contains(':') {
            return Err(TransportConfigError::ProxyCredentialInvalid {
                field: "proxy_username",
            });
        }
        let password = password.into();
        validate_credential_field("proxy_password", &password)?;
        Ok(Self { username, password })
    }

    pub(crate) fn apply_to(&self, proxy: reqwest::Proxy) -> reqwest::Proxy {
        proxy.basic_auth(&self.username, &self.password)
    }
}

impl fmt::Debug for ProxyBasicCredential {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProxyBasicCredential")
            .field("configured", &true)
            .finish()
    }
}

/// The closed provider-HTTP route choice.
#[derive(Clone, Default)]
pub enum HttpTransportRoute {
    /// Connect directly to the provider origin. This is the default.
    #[default]
    Direct,
    /// Establish an explicit CONNECT tunnel through one trusted proxy.
    TrustedConnect(TrustedConnectRoute),
}

#[derive(Clone)]
#[doc(hidden)]
pub struct TrustedConnectRoute {
    proxy: Box<ProxyEndpoint>,
    credential: Option<ProxyBasicCredential>,
}

impl HttpTransportRoute {
    /// Construct a trusted CONNECT route without proxy authentication.
    pub fn trusted_connect(proxy: ProxyEndpoint) -> Self {
        Self::TrustedConnect(TrustedConnectRoute {
            proxy: Box::new(proxy),
            credential: None,
        })
    }

    /// Attach one immutable Basic credential to a trusted CONNECT route.
    pub fn with_basic_auth(
        self,
        credential: ProxyBasicCredential,
    ) -> Result<Self, TransportConfigError> {
        match self {
            Self::Direct => Err(TransportConfigError::ProxyRouteRequired),
            Self::TrustedConnect(route) if !route.proxy.uses_tls() => {
                Err(TransportConfigError::ProxyCredentialsRequireTls)
            }
            Self::TrustedConnect(mut route) => {
                route.credential = Some(credential);
                Ok(Self::TrustedConnect(route))
            }
        }
    }

    /// Return the configured proxy endpoint, if this route uses one.
    pub fn proxy(&self) -> Option<&ProxyEndpoint> {
        match self {
            Self::Direct => None,
            Self::TrustedConnect(route) => Some(&route.proxy),
        }
    }

    /// Return whether this route carries proxy authentication.
    pub fn has_basic_auth(&self) -> bool {
        matches!(self, Self::TrustedConnect(route) if route.credential.is_some())
    }

    /// Build the opaque reqwest proxy configuration for this route.
    ///
    /// This does not expose a client or request builder. It is a narrow
    /// workspace integration seam for HTTP stacks that retain their own
    /// endpoint, authentication, replay, and response-boundary ownership.
    #[doc(hidden)]
    pub fn build_reqwest_proxy(&self) -> Result<Option<reqwest::Proxy>, TransportConfigError> {
        let Self::TrustedConnect(route) = self else {
            return Ok(None);
        };
        if route.credential.is_some() && !route.proxy.uses_tls() {
            return Err(TransportConfigError::ProxyCredentialsRequireTls);
        }
        let proxy_config = reqwest::Proxy::https(route.proxy.url().clone())
            .map_err(|_| TransportConfigError::ClientBuild)?;
        Ok(Some(match &route.credential {
            Some(credential) => credential.apply_to(proxy_config),
            None => proxy_config,
        }))
    }

    pub(crate) fn validate_for_endpoint(
        &self,
        endpoint: &EndpointConfig,
    ) -> Result<(), TransportConfigError> {
        if matches!(self, Self::TrustedConnect(_))
            && (endpoint.expose_base_url().scheme() != "https"
                || matches!(endpoint.policy(), EndpointPolicy::LocalExplicit(_)))
        {
            return Err(TransportConfigError::ProxyDestinationMustBePublicHttps);
        }
        Ok(())
    }
}

impl fmt::Debug for HttpTransportRoute {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct => formatter.write_str("Direct"),
            Self::TrustedConnect(route) => formatter
                .debug_struct("TrustedConnect")
                .field("proxy", &route.proxy)
                .field("credential_configured", &route.credential.is_some())
                .finish(),
        }
    }
}

fn validate_credential_field(field: &'static str, value: &str) -> Result<(), TransportConfigError> {
    if value.is_empty() {
        return Err(TransportConfigError::ProxyCredentialEmpty { field });
    }
    if value.len() > MAX_PROXY_CREDENTIAL_BYTES {
        return Err(TransportConfigError::ProxyCredentialTooLarge {
            field,
            maximum: MAX_PROXY_CREDENTIAL_BYTES,
        });
    }
    if value.chars().any(char::is_control) {
        return Err(TransportConfigError::ProxyCredentialControl { field });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_origin_is_root_only() {
        assert_eq!(
            ProxyEndpoint::https("https://proxy.example.test/path").unwrap_err(),
            EndpointError::ProxyOriginMustBeRoot
        );
        assert_eq!(
            ProxyEndpoint::https("https://proxy.example.test?tenant=secret").unwrap_err(),
            EndpointError::ProxyOriginMustBeRoot
        );
    }

    #[test]
    fn proxy_credentials_are_bounded_and_redacted() {
        assert!(matches!(
            ProxyBasicCredential::new("", "secret"),
            Err(TransportConfigError::ProxyCredentialEmpty {
                field: "proxy_username"
            })
        ));
        assert!(matches!(
            ProxyBasicCredential::new("user\n", "secret"),
            Err(TransportConfigError::ProxyCredentialControl {
                field: "proxy_username"
            })
        ));
        assert!(matches!(
            ProxyBasicCredential::new("user:name", "secret"),
            Err(TransportConfigError::ProxyCredentialInvalid {
                field: "proxy_username"
            })
        ));
        assert!(matches!(
            ProxyBasicCredential::new("u".repeat(MAX_PROXY_CREDENTIAL_BYTES + 1), "secret"),
            Err(TransportConfigError::ProxyCredentialTooLarge {
                field: "proxy_username",
                maximum: MAX_PROXY_CREDENTIAL_BYTES
            })
        ));
        let credential = ProxyBasicCredential::new("user", "secret").unwrap();
        let debug = format!("{credential:?}");
        assert!(!debug.contains("user"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn cleartext_local_proxy_cannot_carry_basic_credentials() {
        let proxy = ProxyEndpoint::local_explicit("http://127.0.0.1:3128").unwrap();
        let route = HttpTransportRoute::trusted_connect(proxy);
        let error = route
            .with_basic_auth(ProxyBasicCredential::new("user", "secret").unwrap())
            .unwrap_err();
        assert_eq!(error, TransportConfigError::ProxyCredentialsRequireTls);
    }

    #[test]
    fn proxy_builder_rechecks_cleartext_credentials() {
        let route = HttpTransportRoute::TrustedConnect(TrustedConnectRoute {
            proxy: Box::new(ProxyEndpoint::local_explicit("http://127.0.0.1:3128").unwrap()),
            credential: Some(ProxyBasicCredential::new("user", "secret").unwrap()),
        });

        assert!(matches!(
            route.build_reqwest_proxy(),
            Err(TransportConfigError::ProxyCredentialsRequireTls)
        ));
    }
}
