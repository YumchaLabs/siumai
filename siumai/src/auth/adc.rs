//! Application Default Credentials (ADC) based Bearer token provider for Google Cloud (Vertex AI).
//!
//! Resolution order (simplified):
//! 1) Environment variable `GOOGLE_OAUTH_ACCESS_TOKEN`
//! 2) Service Account JSON via `GOOGLE_APPLICATION_CREDENTIALS`
//! 3) GCE/GKE metadata server token
//!
//! Tokens are cached in-memory and refreshed before expiration when possible.

use crate::auth::TokenProvider as TokenProviderTrait;
use crate::auth::service_account::{ServiceAccountCredentials, ServiceAccountTokenProvider};
use crate::error::LlmError;
use reqwest::Client;
use serde::Deserialize;
use std::fs;
use std::sync::{Arc, Condvar, Mutex};

const METADATA_URL_DEFAULT: &str =
    "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token";
const METADATA_HEADER: &str = "Metadata-Flavor";
const METADATA_HEADER_VALUE: &str = "Google";
const EXPIRY_SAFETY_WINDOW: i64 = 300; // 5 minutes

#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    exp_unix: i64,
}

#[derive(Debug)]
#[allow(dead_code)]
enum Source {
    Env,
    ServiceAccount,
    Metadata,
}

/// ADC token provider with simple caching.
pub struct AdcTokenProvider {
    http: Client,
    cache: Arc<Mutex<Option<CachedToken>>>,
    // Lazy-initialized SA provider if we resolve SA JSON path
    sa_provider: Arc<Mutex<Option<ServiceAccountTokenProvider>>>,
    // Prevent thundering herd on metadata/SA lookups
    refreshing: Arc<(Mutex<bool>, Condvar)>,
}

impl AdcTokenProvider {
    /// Create a new ADC token provider with a custom HTTP client
    pub fn new(http: Client) -> Self {
        Self {
            http,
            cache: Arc::new(Mutex::new(None)),
            sa_provider: Arc::new(Mutex::new(None)),
            refreshing: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Create a new ADC token provider with default HTTP client
    pub fn default_client() -> Self {
        Self::new(Client::new())
    }

    fn get_cached(&self) -> Option<String> {
        let now = chrono::Utc::now().timestamp();
        if let Ok(g) = self.cache.lock()
            && let Some(ct) = g.as_ref()
            && ct.exp_unix - EXPIRY_SAFETY_WINDOW > now
        {
            return Some(ct.token.clone());
        }
        None
    }

    fn set_cache(&self, token: String, expires_in: i64) {
        let now = chrono::Utc::now().timestamp();
        let exp = now + expires_in;
        if let Ok(mut g) = self.cache.lock() {
            *g = Some(CachedToken {
                token,
                exp_unix: exp,
            });
        }
    }

    fn try_env(&self) -> Option<(String, i64)> {
        if let Ok(tok) = std::env::var("GOOGLE_OAUTH_ACCESS_TOKEN")
            && !tok.is_empty()
        {
            // No expiry info; assume short-lived
            return Some((tok, 600));
        }
        None
    }

    async fn try_service_account(&self) -> Result<Option<(String, i64)>, LlmError> {
        let path = match std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            Ok(p) if !p.is_empty() => p,
            _ => return Ok(None),
        };

        let content = fs::read_to_string(&path).map_err(|e| {
            LlmError::ConfigurationError(format!(
                "Failed to read GOOGLE_APPLICATION_CREDENTIALS file {}: {}",
                path, e
            ))
        })?;
        let creds = ServiceAccountCredentials::from_json(&content)?;

        // Lazy init SA provider and clone it to avoid holding lock across await
        let provider = {
            let mut guard = self.sa_provider.lock().unwrap();
            if guard.is_none() {
                *guard = Some(ServiceAccountTokenProvider::new(
                    creds,
                    self.http.clone(),
                    None,
                ));
            }
            guard.clone()
        };

        if let Some(p) = provider {
            let tok = p.token().await?;
            // We don't have exact expiry; assume one hour since SA provider uses 3600s
            return Ok(Some((tok, 3600)));
        }

        Ok(None)
    }

    async fn try_metadata(&self) -> Result<Option<(String, i64)>, LlmError> {
        // Allow test override via ADC_METADATA_URL
        let url =
            std::env::var("ADC_METADATA_URL").unwrap_or_else(|_| METADATA_URL_DEFAULT.to_string());
        let resp = self
            .http
            .get(url)
            .header(METADATA_HEADER, METADATA_HEADER_VALUE)
            .send()
            .await
            .map_err(|e| LlmError::HttpError(format!("Metadata server request failed: {e}")))?;
        if !resp.status().is_success() {
            return Ok(None);
        }
        #[derive(Deserialize)]
        struct MdResp {
            access_token: String,
            expires_in: i64,
        }
        let m: MdResp = resp.json().await.map_err(|e| {
            LlmError::ParseError(format!("Failed to parse metadata token response: {e}"))
        })?;
        Ok(Some((m.access_token, m.expires_in)))
    }
}

#[async_trait::async_trait]
impl TokenProviderTrait for AdcTokenProvider {
    async fn token(&self) -> Result<String, LlmError> {
        if let Some(t) = self.get_cached() {
            return Ok(t);
        }
        // Single-flight refresh
        let (lock, cvar) = &*self.refreshing;

        // Check if we should refresh
        let should_refresh = {
            let mut refreshing = lock.lock().unwrap();
            if !*refreshing {
                *refreshing = true;
                true
            } else {
                false
            }
        };

        if should_refresh {
            // Resolve in order
            if let Some((t, exp)) = self.try_env() {
                self.set_cache(t.clone(), exp);
                let mut refreshing = lock.lock().unwrap();
                *refreshing = false;
                cvar.notify_all();
                return Ok(t);
            }
            if let Some((t, exp)) = self.try_service_account().await? {
                self.set_cache(t.clone(), exp);
                let mut refreshing = lock.lock().unwrap();
                *refreshing = false;
                cvar.notify_all();
                return Ok(t);
            }
            if let Some((t, exp)) = self.try_metadata().await? {
                self.set_cache(t.clone(), exp);
                let mut refreshing = lock.lock().unwrap();
                *refreshing = false;
                cvar.notify_all();
                return Ok(t);
            }
            let mut refreshing = lock.lock().unwrap();
            *refreshing = false;
            cvar.notify_all();
            return Err(LlmError::ConfigurationError(
                "ADC resolution failed: no env token, no service account file, no metadata token"
                    .to_string(),
            ));
        }

        // Wait for in-flight resolution
        let mut refreshing = lock.lock().unwrap();
        while *refreshing {
            refreshing = cvar.wait(refreshing).unwrap();
        }
        if let Some(t) = self.get_cached() {
            Ok(t)
        } else {
            Err(LlmError::ConfigurationError(
                "ADC resolution failed after refresh".to_string(),
            ))
        }
    }
}
