//! Client utilities for Siumai (extras crate)
//!
//! This module hosts helper types for managing and pooling `siumai::experimental::client::ClientWrapper`
//! instances. These utilities were originally part of the core crate and are now
//! considered application-level helpers.

use siumai::experimental::client::ClientWrapper;
use siumai::prelude::unified::{CommonParams, HttpConfig};

/// Client configuration for advanced client setup.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// API Key for authentication
    pub api_key: String,
    /// Base URL for the provider API
    pub base_url: String,
    /// HTTP Configuration (timeouts, retries, etc.)
    pub http_config: HttpConfig,
    /// Common Parameters (temperature, max_tokens, etc.)
    pub common_params: CommonParams,
}

impl ClientConfig {
    /// Creates a new client configuration
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            api_key,
            base_url,
            http_config: HttpConfig::default(),
            common_params: CommonParams::default(),
        }
    }

    /// Sets the HTTP configuration
    pub fn with_http_config(mut self, config: HttpConfig) -> Self {
        self.http_config = config;
        self
    }

    /// Sets the common parameters
    pub fn with_common_params(mut self, params: CommonParams) -> Self {
        self.common_params = params;
        self
    }
}

/// Client Manager - manage multiple named client instances.
pub struct ClientManager {
    clients: std::collections::HashMap<String, ClientWrapper>,
}

impl ClientManager {
    /// Creates a new client manager
    pub fn new() -> Self {
        Self {
            clients: std::collections::HashMap::new(),
        }
    }

    /// Adds a client
    pub fn add_client(&mut self, name: String, client: ClientWrapper) {
        self.clients.insert(name, client);
    }

    /// Gets a client
    pub fn get_client(&self, name: &str) -> Option<&ClientWrapper> {
        self.clients.get(name)
    }

    /// Removes a client
    pub fn remove_client(&mut self, name: &str) -> Option<ClientWrapper> {
        self.clients.remove(name)
    }

    /// Lists all client names
    pub fn list_clients(&self) -> Vec<&String> {
        self.clients.keys().collect()
    }

    /// Gets the default client (the first one added)
    pub fn default_client(&self) -> Option<&ClientWrapper> {
        self.clients.values().next()
    }
}

impl Default for ClientManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Client Pool - used for connection pool management.
pub struct ClientPool {
    pool: std::sync::Arc<std::sync::Mutex<Vec<ClientWrapper>>>,
    max_size: usize,
}

impl ClientPool {
    /// Creates a new client pool
    pub fn new(max_size: usize) -> Self {
        Self {
            pool: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            max_size,
        }
    }

    /// Gets a client
    pub fn get_client(&self) -> Option<ClientWrapper> {
        let mut pool = self.pool.lock().unwrap();
        pool.pop()
    }

    /// Returns a client
    pub fn return_client(&self, client: ClientWrapper) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.max_size {
            pool.push(client);
        }
    }

    /// Gets the pool size
    pub fn size(&self) -> usize {
        let pool = self.pool.lock().unwrap();
        pool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_client_manager() {
        let manager = ClientManager::new();
        assert_eq!(manager.list_clients().len(), 0);
        assert!(manager.default_client().is_none());
    }

    #[test]
    fn test_client_pool() {
        let pool = ClientPool::new(5);
        assert_eq!(pool.size(), 0);
        assert!(pool.get_client().is_none());
    }

    #[test]
    fn test_client_config() {
        let config = ClientConfig::new(
            "test-key".to_string(),
            "https://api.example.com".to_string(),
        );
        assert_eq!(config.api_key, "test-key");
        assert_eq!(config.base_url, "https://api.example.com");
    }

    // Test that client types are Send + Sync for multi-threading
    #[test]
    fn test_client_types_are_send_sync() {
        fn test_arc_usage() {
            let _: Option<Arc<ClientWrapper>> = None;
            let _: Option<Arc<ClientManager>> = None;
            let _: Option<Arc<ClientPool>> = None;
        }

        test_arc_usage();
    }

    // Test multi-threading with ClientPool
    #[tokio::test]
    async fn test_client_pool_multithreading() {
        use tokio::task;

        let pool = Arc::new(ClientPool::new(5));

        // Spawn multiple tasks that access the pool concurrently
        let mut handles = Vec::new();

        for i in 0..10 {
            let pool_clone = pool.clone();
            let handle = task::spawn(async move {
                // Try to get a client (will be None since pool is empty)
                let client = pool_clone.get_client();
                assert!(client.is_none());

                // Check pool size
                let size = pool_clone.size();
                assert_eq!(size, 0);

                i // Return task id for verification
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        let mut results = Vec::new();
        for handle in handles {
            let result = handle.await.unwrap();
            results.push(result);
        }

        // Verify all tasks completed
        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(*result, i);
        }
    }
}
