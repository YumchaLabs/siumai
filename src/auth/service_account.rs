//! Service Account based Bearer token provider for Google Cloud (Vertex AI).
//!
//! This provider implements the OAuth 2.0 JWT Bearer grant flow using a Service Account
//! private key to obtain an access token from Google's token endpoint. Tokens are cached
//! in-memory and refreshed before expiration.

use crate::error::LlmError;
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Condvar, Mutex};

/// Default Google OAuth token endpoint
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
/// Default scope for Vertex AI access (full Cloud Platform)
const DEFAULT_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
/// Safety window (seconds) to refresh before expiry
const EXPIRY_SAFETY_WINDOW: i64 = 300; // 5 minutes

/// Service Account credential subset (fields required for JWT flow)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAccountCredentials {
    pub client_email: String,
    pub private_key: String,
    #[serde(default)]
    pub token_uri: Option<String>,
    /// Optional OAuth scopes; if empty, defaults to cloud-platform.
    #[serde(default)]
    pub scopes: Vec<String>,
}

impl ServiceAccountCredentials {
    /// Create from raw fields.
    pub fn new(client_email: String, private_key: String) -> Self {
        Self {
            client_email,
            private_key,
            token_uri: None,
            scopes: vec![],
        }
    }

    /// Create from a Service Account JSON string.
    pub fn from_json(json: &str) -> Result<Self, LlmError> {
        serde_json::from_str::<Self>(json)
            .map_err(|e| LlmError::ConfigurationError(format!("Invalid service account JSON: {e}")))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Claims {
    iss: String,
    scope: String,
    aud: String,
    exp: i64,
    iat: i64,
    sub: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenResponse {
    access_token: String,
    token_type: String,
    expires_in: i64,
}

#[derive(Debug, Clone)]
struct CachedToken {
    token: String,
    /// Unix timestamp seconds when token expires
    exp_unix: i64,
}

/// Service Account based token provider with in-memory caching.
pub struct ServiceAccountTokenProvider {
    creds: ServiceAccountCredentials,
    http: reqwest::blocking::Client,
    cache: Arc<Mutex<Option<CachedToken>>>,
    subject: Option<String>,
    assertion_override: Option<String>,
    // Prevent thundering herd on concurrent refresh
    refreshing: Arc<(Mutex<bool>, Condvar)>,
}

impl ServiceAccountTokenProvider {
    /// Create a new provider.
    /// - `http` can be a shared reqwest client.
    /// - `subject` is optional user to impersonate (rarely needed).
    pub fn new(
        creds: ServiceAccountCredentials,
        http: reqwest::blocking::Client,
        subject: Option<String>,
    ) -> Self {
        Self {
            creds,
            http,
            cache: Arc::new(Mutex::new(None)),
            subject,
            assertion_override: None,
            refreshing: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    /// Constructor to bypass cryptographic signing and inject a prebuilt assertion (primarily for tests).
    pub fn new_with_assertion_override(
        creds: ServiceAccountCredentials,
        http: reqwest::blocking::Client,
        subject: Option<String>,
        assertion: String,
    ) -> Self {
        Self {
            creds,
            http,
            cache: Arc::new(Mutex::new(None)),
            subject,
            assertion_override: Some(assertion),
            refreshing: Arc::new((Mutex::new(false), Condvar::new())),
        }
    }

    fn scope_string(&self) -> String {
        if self.creds.scopes.is_empty() {
            DEFAULT_SCOPE.to_string()
        } else {
            self.creds.scopes.join(" ")
        }
    }

    fn token_uri(&self) -> String {
        self.creds
            .token_uri
            .clone()
            .unwrap_or_else(|| DEFAULT_TOKEN_URI.to_string())
    }

    /// Returns a cached token if valid and not near expiry.
    fn get_cached_token(&self) -> Option<String> {
        let now = chrono::Utc::now().timestamp();
        if let Ok(guard) = self.cache.lock()
            && let Some(ct) = guard.as_ref()
            && ct.exp_unix - EXPIRY_SAFETY_WINDOW > now
        {
            return Some(ct.token.clone());
        }
        None
    }

    fn cache_token(&self, token: String, expires_in: i64) {
        let now = chrono::Utc::now().timestamp();
        let exp_unix = now + expires_in;
        if let Ok(mut guard) = self.cache.lock() {
            *guard = Some(CachedToken { token, exp_unix });
        }
    }

    /// Perform JWT Bearer grant to obtain a new access token.
    fn fetch_new_token(&self) -> Result<String, LlmError> {
        // Build JWT claims
        let now = chrono::Utc::now().timestamp();
        let scope = self.scope_string();
        let aud = self.token_uri();
        let claims = Claims {
            iss: self.creds.client_email.clone(),
            scope,
            aud: aud.clone(),
            iat: now,
            exp: now + 3600,
            sub: self.subject.clone(),
        };

        let assertion = if let Some(a) = &self.assertion_override {
            a.clone()
        } else {
            let mut header = Header::new(Algorithm::RS256);
            header.typ = Some("JWT".to_string());
            let key =
                EncodingKey::from_rsa_pem(self.creds.private_key.as_bytes()).map_err(|e| {
                    LlmError::ConfigurationError(format!("Invalid RSA private key (PEM): {e}"))
                })?;
            encode(&header, &claims, &key)
                .map_err(|e| LlmError::ConfigurationError(format!("Failed to sign JWT: {e}")))?
        };

        // Exchange JWT for access token
        let form = [
            ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
            ("assertion", assertion.as_str()),
        ];

        let resp = self
            .http
            .post(aud)
            .form(&form)
            .send()
            .map_err(|e| LlmError::HttpError(format!("Token endpoint request failed: {e}")))?;

        let resp = resp
            .error_for_status()
            .map_err(|e| LlmError::HttpError(format!("Token endpoint returned error: {e}")))?;

        let tr: TokenResponse = resp
            .json()
            .map_err(|e| LlmError::ParseError(format!("Failed to parse token response: {e}")))?;

        self.cache_token(tr.access_token.clone(), tr.expires_in);
        Ok(tr.access_token)
    }
}

impl crate::auth::TokenProvider for ServiceAccountTokenProvider {
    fn token(&self) -> Result<String, LlmError> {
        if let Some(tok) = self.get_cached_token() {
            return Ok(tok);
        }
        // Prevent thundering herd: single fetch, others wait
        let (lock, cvar) = &*self.refreshing;
        // Fast path: attempt to become the refesher
        {
            let mut refreshing = lock.lock().unwrap();
            if !*refreshing {
                *refreshing = true;
                drop(refreshing);
                let res = self.fetch_new_token();
                let mut refreshing = lock.lock().unwrap();
                *refreshing = false;
                cvar.notify_all();
                return res;
            }
        }
        // Wait for the in-flight refresh to complete
        let mut refreshing = lock.lock().unwrap();
        while *refreshing {
            refreshing = cvar.wait(refreshing).unwrap();
        }
        if let Some(tok) = self.get_cached_token() {
            return Ok(tok);
        }
        // As a fallback, fetch again (rare race)
        *refreshing = true;
        drop(refreshing);
        let res = self.fetch_new_token();
        let mut refreshing = lock.lock().unwrap();
        *refreshing = false;
        cvar.notify_all();
        res
    }
}
