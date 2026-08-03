//! Endpoint policy, exact credential audiences, and DNS pinning.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use async_trait::async_trait;
use reqwest::Url;
use url::Host;

use crate::{EndpointError, RequestTarget};

/// A validated, provider-owned production origin.
///
/// Provider adapters should create this value from their audited endpoint
/// metadata and retain it independently from caller-supplied base URL
/// overrides. Its private representation prevents unchecked construction; an
/// [`EndpointConfig`] using the official policy must match this exact origin.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct OfficialOrigin {
    audience: CredentialAudience,
}

impl OfficialOrigin {
    /// Validate an exact HTTPS production origin declared by a provider.
    ///
    /// The input must contain only a scheme, host, and optional port. Paths,
    /// queries, fragments, user information, local names, and private address
    /// literals are rejected. DNS answers remain subject to connector-time
    /// validation when the endpoint is used.
    ///
    /// This validates and seals the declared origin; it does not discover
    /// vendor ownership. Provider adapters must source the declaration from
    /// audited provider metadata, never from a caller's endpoint override.
    pub fn new(origin: impl AsRef<str>) -> Result<Self, EndpointError> {
        let origin =
            Url::parse(origin.as_ref()).map_err(|_| EndpointError::InvalidOfficialOrigin)?;
        validate_official_origin_shape(&origin)?;
        if let Some(address) = literal_address(&origin)? {
            validate_address(address, &EndpointPolicy::PublicCustom)
                .map_err(|_| EndpointError::InvalidOfficialOrigin)?;
        }
        Ok(Self {
            audience: CredentialAudience::from_url(&origin)
                .map_err(|_| EndpointError::InvalidOfficialOrigin)?,
        })
    }

    /// Return the exact credential audience represented by this origin.
    pub fn audience(&self) -> &CredentialAudience {
        &self.audience
    }
}

impl fmt::Debug for OfficialOrigin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OfficialOrigin")
            .field("origin", &"[REDACTED]")
            .finish()
    }
}

/// Security policy for one configured remote origin.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EndpointPolicy {
    /// A provider-owned endpoint bound to the supplied exact production origin.
    Official(OfficialOrigin),
    /// A caller-supplied production endpoint. HTTPS and public targets are required.
    PublicCustom,
    /// An explicitly selected local-network scope.
    LocalExplicit(LocalNetworkGrant),
}

/// Exact non-public network scope authorized by a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LocalNetworkGrant {
    /// IPv4 or IPv6 loopback only.
    Loopback,
    /// RFC 1918 IPv4 or IPv6 unique-local addresses only.
    PrivateNetwork,
    /// IPv4 or IPv6 link-local addresses only. This includes metadata ranges.
    LinkLocal,
}

impl EndpointPolicy {
    /// Return whether the policy requires TLS for provider requests.
    pub fn requires_tls(&self) -> bool {
        matches!(self, Self::Official(_) | Self::PublicCustom)
    }

    /// Return whether every connection target must be globally routable.
    pub fn requires_public_network(&self) -> bool {
        matches!(self, Self::Official(_) | Self::PublicCustom)
    }

    /// Return the provider-owned origin proof for an official policy.
    pub fn official_origin(&self) -> Option<&OfficialOrigin> {
        match self {
            Self::Official(origin) => Some(origin),
            Self::PublicCustom | Self::LocalExplicit(_) => None,
        }
    }
}

/// Exact scheme, host, and effective port to which credentials are bound.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CredentialAudience {
    scheme: String,
    host: String,
    port: u16,
}

impl CredentialAudience {
    pub fn from_url(url: &Url) -> Result<Self, EndpointError> {
        let host = url.host_str().ok_or(EndpointError::MissingHost)?;
        let port = url
            .port_or_known_default()
            .ok_or(EndpointError::SchemeNotAllowed)?;
        Ok(Self {
            scheme: url.scheme().to_ascii_lowercase(),
            host: host.to_ascii_lowercase(),
            port,
        })
    }

    pub fn matches(&self, url: &Url) -> bool {
        Self::from_url(url).is_ok_and(|candidate| candidate == *self)
    }

    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

impl fmt::Debug for CredentialAudience {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CredentialAudience")
            .field("scheme", &self.scheme)
            .field("host", &"[REDACTED]")
            .field("port", &self.port)
            .finish()
    }
}

/// Statically validated endpoint settings. DNS is guarded by the connector.
#[derive(Clone)]
pub struct EndpointConfig {
    base_url: Url,
    policy: EndpointPolicy,
    audience: CredentialAudience,
}

impl EndpointConfig {
    /// Construct an endpoint bound to provider-owned official origin metadata.
    pub fn official(
        base_url: impl AsRef<str>,
        origin: OfficialOrigin,
    ) -> Result<Self, EndpointError> {
        Self::new(base_url, EndpointPolicy::Official(origin))
    }

    /// Construct a caller-supplied HTTPS endpoint that resolves publicly.
    pub fn public_custom(base_url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(base_url, EndpointPolicy::PublicCustom)
    }

    /// Construct an explicitly authorized HTTP(S) endpoint on a local network.
    /// This convenience path grants loopback access only.
    pub fn local_explicit(base_url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            base_url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::Loopback),
        )
    }

    /// Construct an endpoint explicitly authorized for RFC 1918 or IPv6 ULA targets.
    pub fn private_network_explicit(base_url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            base_url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::PrivateNetwork),
        )
    }

    /// Construct an endpoint explicitly authorized for link-local targets.
    pub fn link_local_explicit(base_url: impl AsRef<str>) -> Result<Self, EndpointError> {
        Self::new(
            base_url,
            EndpointPolicy::LocalExplicit(LocalNetworkGrant::LinkLocal),
        )
    }

    /// Construct an endpoint from an explicit security policy.
    pub fn new(base_url: impl AsRef<str>, policy: EndpointPolicy) -> Result<Self, EndpointError> {
        let base_url = Url::parse(base_url.as_ref()).map_err(|_| EndpointError::InvalidUrl)?;
        validate_url_shape(&base_url, &policy)?;
        if let Some(address) = literal_address(&base_url)? {
            validate_address(address, &policy)?;
        }
        let audience = CredentialAudience::from_url(&base_url)?;
        if let EndpointPolicy::Official(origin) = &policy
            && origin.audience() != &audience
        {
            return Err(EndpointError::OfficialOriginMismatch);
        }
        Ok(Self {
            base_url,
            policy,
            audience,
        })
    }

    pub fn policy(&self) -> EndpointPolicy {
        self.policy.clone()
    }

    pub fn audience(&self) -> &CredentialAudience {
        &self.audience
    }

    /// Explicit access to the configured URL. Avoid logging it because its
    /// query can contain signed provider parameters.
    pub fn expose_base_url(&self) -> &Url {
        &self.base_url
    }

    pub fn request_url(&self, target: &RequestTarget) -> Result<Url, EndpointError> {
        let inherited_query = self.base_url.query().map(str::to_owned);
        let mut base = self.base_url.clone();
        base.set_query(None);
        if !base.path().ends_with('/') {
            let mut path = base.path().to_owned();
            path.push('/');
            base.set_path(&path);
        }

        let mut url = base
            .join(target.as_str().trim_start_matches('/'))
            .map_err(|_| EndpointError::InvalidUrl)?;
        if !self.audience.matches(&url) {
            return Err(EndpointError::AudienceMismatch);
        }
        if let Some(inherited_query) = inherited_query {
            let merged = match url.query() {
                Some(target_query) if !target_query.is_empty() => {
                    format!("{inherited_query}&{target_query}")
                }
                _ => inherited_query,
            };
            url.set_query(Some(&merged));
        }
        Ok(url)
    }

    pub(crate) async fn validated_addresses<R: Resolver + ?Sized>(
        &self,
        resolver: &R,
    ) -> Result<Vec<SocketAddr>, EndpointError> {
        let port = self.audience.port;
        let addresses = match self.base_url.host().ok_or(EndpointError::MissingHost)? {
            Host::Ipv4(address) => vec![SocketAddr::new(IpAddr::V4(address), port)],
            Host::Ipv6(address) => vec![SocketAddr::new(IpAddr::V6(address), port)],
            Host::Domain(host) => resolver.resolve(host, port).await?,
        };
        if addresses.is_empty() {
            return Err(EndpointError::NoAddresses);
        }
        for address in &addresses {
            validate_address(address.ip(), &self.policy)?;
        }
        Ok(addresses)
    }

    pub(crate) async fn resolve_for_connector<R: Resolver + ?Sized>(
        &self,
        requested_host: &str,
        resolver: &R,
    ) -> Result<Vec<SocketAddr>, EndpointError> {
        let expected = match self.base_url.host().ok_or(EndpointError::MissingHost)? {
            Host::Domain(host) => host,
            Host::Ipv4(_) | Host::Ipv6(_) => return Err(EndpointError::HostNotAllowed),
        };
        if !expected.eq_ignore_ascii_case(requested_host) {
            return Err(EndpointError::HostNotAllowed);
        }
        let mut addresses = resolver.resolve(requested_host, 0).await?;
        if addresses.is_empty() {
            return Err(EndpointError::NoAddresses);
        }
        for address in &mut addresses {
            validate_address(address.ip(), &self.policy)?;
            address.set_port(0);
        }
        addresses.sort_unstable();
        addresses.dedup();
        Ok(addresses)
    }

    pub(crate) fn validate_remote(&self, remote: SocketAddr) -> Result<(), EndpointError> {
        if remote.port() != self.audience.port {
            return Err(EndpointError::AddressNotAllowed);
        }
        validate_address(remote.ip(), &self.policy)?;
        match self.base_url.host().ok_or(EndpointError::MissingHost)? {
            Host::Ipv4(expected) if remote.ip() != IpAddr::V4(expected) => {
                Err(EndpointError::AddressNotAllowed)
            }
            Host::Ipv6(expected) if remote.ip() != IpAddr::V6(expected) => {
                Err(EndpointError::AddressNotAllowed)
            }
            Host::Domain(_) | Host::Ipv4(_) | Host::Ipv6(_) => Ok(()),
        }
    }
}

impl fmt::Debug for EndpointConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EndpointConfig")
            .field("policy", &self.policy)
            .field("audience", &self.audience)
            .field("base_url", &"[REDACTED]")
            .finish()
    }
}

/// DNS abstraction used by endpoint validation and deterministic tests.
#[async_trait]
pub trait Resolver: Send + Sync {
    async fn resolve(&self, host: &str, port: u16) -> Result<Vec<SocketAddr>, EndpointError>;
}

/// Tokio's system resolver.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemResolver;

#[async_trait]
impl Resolver for SystemResolver {
    async fn resolve(&self, host: &str, port: u16) -> Result<Vec<SocketAddr>, EndpointError> {
        let mut addresses = tokio::net::lookup_host((host, port))
            .await
            .map_err(|_| EndpointError::ResolutionFailed)?
            .collect::<Vec<_>>();
        addresses.sort_unstable();
        addresses.dedup();
        Ok(addresses)
    }
}

fn validate_official_origin_shape(url: &Url) -> Result<(), EndpointError> {
    if url.path() != "/" || url.query().is_some() {
        return Err(EndpointError::InvalidOfficialOrigin);
    }
    validate_url_shape(url, &EndpointPolicy::PublicCustom)
        .map_err(|_| EndpointError::InvalidOfficialOrigin)
}

fn literal_address(url: &Url) -> Result<Option<IpAddr>, EndpointError> {
    match url.host().ok_or(EndpointError::MissingHost)? {
        Host::Ipv4(address) => Ok(Some(IpAddr::V4(address))),
        Host::Ipv6(address) => Ok(Some(IpAddr::V6(address))),
        Host::Domain(_) => Ok(None),
    }
}

fn validate_url_shape(url: &Url, policy: &EndpointPolicy) -> Result<(), EndpointError> {
    match policy {
        EndpointPolicy::Official(_) | EndpointPolicy::PublicCustom if url.scheme() != "https" => {
            return Err(EndpointError::SchemeNotAllowed);
        }
        EndpointPolicy::LocalExplicit(_) if !matches!(url.scheme(), "http" | "https") => {
            return Err(EndpointError::SchemeNotAllowed);
        }
        _ => {}
    }
    if url.host_str().is_none() {
        return Err(EndpointError::MissingHost);
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(EndpointError::UserInfoNotAllowed);
    }
    if url.fragment().is_some() {
        return Err(EndpointError::FragmentNotAllowed);
    }

    let original_host = url.host_str().expect("host presence was checked");
    if original_host.ends_with('.') {
        return Err(EndpointError::HostNotAllowed);
    }
    let host = original_host.to_ascii_lowercase();
    let local_name =
        host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local");
    match policy {
        EndpointPolicy::Official(_) | EndpointPolicy::PublicCustom if local_name => {
            Err(EndpointError::HostNotAllowed)
        }
        EndpointPolicy::LocalExplicit(_) if !local_name && host.parse::<IpAddr>().is_err() => {
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_address(address: IpAddr, policy: &EndpointPolicy) -> Result<(), EndpointError> {
    let allowed = match policy {
        EndpointPolicy::Official(_) | EndpointPolicy::PublicCustom => is_public_ip(address),
        EndpointPolicy::LocalExplicit(grant) => local_grant_allows(address, *grant),
    };
    if allowed {
        Ok(())
    } else {
        Err(EndpointError::AddressNotAllowed)
    }
}

fn is_public_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => is_public_ipv4(address),
        IpAddr::V6(address) => is_public_ipv6(address),
    }
}

fn is_public_ipv4(address: Ipv4Addr) -> bool {
    let [a, b, c, _] = address.octets();
    !(a == 0
        || a == 10
        || a == 127
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 168)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || a >= 224)
}

fn is_public_ipv6(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4_mapped() {
        return is_public_ipv4(mapped);
    }
    let segments = address.segments();
    let global_unicast = (segments[0] & 0xe000) == 0x2000;
    let special_2001 = segments[0] == 0x2001 && segments[1] <= 0x01ff;
    let documentation = segments[0] == 0x2001 && segments[1] == 0x0db8;
    let transition_or_obsolete = matches!(segments[0], 0x2002 | 0x3ffe);
    global_unicast && !special_2001 && !documentation && !transition_or_obsolete
}

fn local_grant_allows(address: IpAddr, grant: LocalNetworkGrant) -> bool {
    match grant {
        LocalNetworkGrant::Loopback => is_loopback_ip(address),
        LocalNetworkGrant::PrivateNetwork => is_private_ip(address),
        LocalNetworkGrant::LinkLocal => is_link_local_ip(address),
    }
}

fn is_loopback_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => address.is_loopback(),
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return mapped.is_loopback();
            }
            address.is_loopback()
        }
    }
}

fn is_private_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => address.is_private(),
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return mapped.is_private();
            }
            (address.segments()[0] & 0xfe00) == 0xfc00
        }
    }
}

fn is_link_local_ip(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => address.is_link_local(),
        IpAddr::V6(address) => {
            if let Some(mapped) = address.to_ipv4_mapped() {
                return mapped.is_link_local();
            }
            (address.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedResolver(Vec<SocketAddr>);

    #[async_trait]
    impl Resolver for FixedResolver {
        async fn resolve(&self, _host: &str, _port: u16) -> Result<Vec<SocketAddr>, EndpointError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn public_endpoint_rejects_credentials_and_fragments() {
        assert_eq!(
            EndpointConfig::new(
                "https://user:secret@example.com/v1",
                EndpointPolicy::PublicCustom
            )
            .unwrap_err(),
            EndpointError::UserInfoNotAllowed
        );
        assert_eq!(
            EndpointConfig::new(
                "https://example.com/v1#secret",
                EndpointPolicy::PublicCustom
            )
            .unwrap_err(),
            EndpointError::FragmentNotAllowed
        );
    }

    #[tokio::test]
    async fn public_endpoint_rejects_any_private_dns_answer() {
        let endpoint =
            EndpointConfig::new("https://example.com/v1", EndpointPolicy::PublicCustom).unwrap();
        let resolver = FixedResolver(vec![
            "93.184.216.34:443".parse().unwrap(),
            "127.0.0.1:443".parse().unwrap(),
        ]);
        assert_eq!(
            endpoint.validated_addresses(&resolver).await.unwrap_err(),
            EndpointError::AddressNotAllowed
        );
    }

    #[tokio::test]
    async fn local_endpoint_requires_a_local_connection_target() {
        let endpoint = EndpointConfig::local_explicit("http://localhost:8080/v1").unwrap();
        let public = FixedResolver(vec!["93.184.216.34:8080".parse().unwrap()]);
        assert_eq!(
            endpoint.validated_addresses(&public).await.unwrap_err(),
            EndpointError::AddressNotAllowed
        );
        let local = FixedResolver(vec!["127.0.0.1:8080".parse().unwrap()]);
        endpoint.validated_addresses(&local).await.unwrap();
    }

    #[test]
    fn local_network_scopes_require_separate_explicit_grants() {
        assert_eq!(
            EndpointConfig::local_explicit("http://10.0.0.7:8080/v1").unwrap_err(),
            EndpointError::AddressNotAllowed
        );
        EndpointConfig::private_network_explicit("http://10.0.0.7:8080/v1").unwrap();
        assert_eq!(
            EndpointConfig::private_network_explicit("http://169.254.169.254/v1").unwrap_err(),
            EndpointError::AddressNotAllowed
        );
        EndpointConfig::link_local_explicit("http://169.254.169.254/v1").unwrap();
        assert_eq!(
            EndpointConfig::link_local_explicit("http://127.0.0.1:8080/v1").unwrap_err(),
            EndpointError::AddressNotAllowed
        );
    }

    #[test]
    fn request_target_cannot_change_credential_audience() {
        let origin = OfficialOrigin::new("https://api.example.com").unwrap();
        let endpoint =
            EndpointConfig::official("https://api.example.com/v1?api-version=next", origin)
                .unwrap();
        let target = RequestTarget::new("responses?stream=true").unwrap();
        let url = endpoint.request_url(&target).unwrap();
        assert_eq!(url.path(), "/v1/responses");
        assert_eq!(url.query(), Some("api-version=next&stream=true"));
        assert!(endpoint.audience().matches(&url));
    }

    #[test]
    fn official_endpoint_requires_the_provider_owned_exact_origin() {
        let origin = OfficialOrigin::new("https://api.example.com").unwrap();
        EndpointConfig::official("https://api.example.com/v1", origin.clone()).unwrap();

        assert_eq!(
            EndpointConfig::official("https://uploads.example.com/v1", origin.clone()).unwrap_err(),
            EndpointError::OfficialOriginMismatch
        );
        assert_eq!(
            EndpointConfig::official("https://api.example.com:8443/v1", origin).unwrap_err(),
            EndpointError::OfficialOriginMismatch
        );
    }

    #[test]
    fn official_origin_rejects_non_origin_and_local_inputs() {
        for candidate in [
            "https://api.example.com/v1",
            "https://api.example.com?key=secret",
            "https://user:secret@api.example.com",
            "http://api.example.com",
            "https://localhost",
            "https://127.0.0.1",
        ] {
            assert_eq!(
                OfficialOrigin::new(candidate).unwrap_err(),
                EndpointError::InvalidOfficialOrigin
            );
        }
    }

    #[test]
    fn endpoint_debug_surfaces_do_not_expose_the_origin() {
        let canary = "canary-provider-origin.example";
        let origin = OfficialOrigin::new(format!("https://{canary}")).unwrap();
        let endpoint = EndpointConfig::official(
            format!("https://{canary}/v1?credential=canary-secret"),
            origin.clone(),
        )
        .unwrap();

        for surface in [
            format!("{origin:?}"),
            format!("{:?}", endpoint.policy()),
            format!("{:?}", endpoint.audience()),
            format!("{endpoint:?}"),
        ] {
            assert!(!surface.contains(canary));
            assert!(!surface.contains("canary-secret"));
        }
    }

    #[test]
    fn mapped_loopback_is_not_public() {
        let mapped = "::ffff:127.0.0.1".parse::<Ipv6Addr>().unwrap();
        assert!(!is_public_ipv6(mapped));
    }

    #[tokio::test]
    async fn ipv6_literal_uses_literal_address_without_dns() {
        let endpoint = EndpointConfig::local_explicit("http://[::1]:8080/v1").unwrap();
        let resolver = FixedResolver(Vec::new());
        let resolved = endpoint.validated_addresses(&resolver).await.unwrap();
        assert_eq!(resolved.as_slice(), &["[::1]:8080".parse().unwrap()]);
    }

    #[test]
    fn trailing_dot_is_not_folded_into_an_audience() {
        assert_eq!(
            EndpointConfig::new("https://example.com./v1", EndpointPolicy::PublicCustom)
                .unwrap_err(),
            EndpointError::HostNotAllowed
        );
    }
}
