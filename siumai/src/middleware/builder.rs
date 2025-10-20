//! Middleware builder for flexible middleware chain construction.

use super::{LanguageModelMiddleware, NamedMiddleware};
use std::sync::Arc;

/// Builder for constructing middleware chains with flexible manipulation.
///
/// Provides a fluent API for adding, removing, replacing, and inserting middleware
/// by name, making it easy to customize the middleware chain.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::middleware::MiddlewareBuilder;
/// use std::sync::Arc;
///
/// let mut builder = MiddlewareBuilder::new();
/// builder
///     .add("logging", Arc::new(LoggingMiddleware))
///     .add("caching", Arc::new(CachingMiddleware))
///     .insert_after("logging", "metrics", Arc::new(MetricsMiddleware))
///     .remove("caching");
///
/// let middlewares = builder.build();
/// ```
#[derive(Default)]
pub struct MiddlewareBuilder {
    middlewares: Vec<NamedMiddleware>,
}

impl MiddlewareBuilder {
    /// Create a new empty middleware builder.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a named middleware to the end of the chain.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this middleware
    /// * `middleware` - The middleware implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.add("logging", Arc::new(LoggingMiddleware));
    /// ```
    pub fn add(
        &mut self,
        name: impl Into<String>,
        middleware: Arc<dyn LanguageModelMiddleware>,
    ) -> &mut Self {
        self.middlewares
            .push(NamedMiddleware::new(name, middleware));
        self
    }

    /// Add a named middleware (takes ownership of NamedMiddleware).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let named = NamedMiddleware::new("logging", Arc::new(LoggingMiddleware));
    /// builder.add_named(named);
    /// ```
    pub fn add_named(&mut self, named: NamedMiddleware) -> &mut Self {
        self.middlewares.push(named);
        self
    }

    /// Insert a middleware before a specific named middleware.
    ///
    /// If the target middleware is not found, logs a warning and does nothing.
    ///
    /// # Arguments
    ///
    /// * `target_name` - Name of the middleware to insert before
    /// * `name` - Name for the new middleware
    /// * `middleware` - The middleware implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.insert_before("caching", "logging", Arc::new(LoggingMiddleware));
    /// ```
    pub fn insert_before(
        &mut self,
        target_name: &str,
        name: impl Into<String>,
        middleware: Arc<dyn LanguageModelMiddleware>,
    ) -> &mut Self {
        if let Some(index) = self.middlewares.iter().position(|m| m.name == target_name) {
            self.middlewares
                .insert(index, NamedMiddleware::new(name, middleware));
        } else {
            tracing::warn!(
                "MiddlewareBuilder: Middleware named '{}' not found, cannot insert before",
                target_name
            );
        }
        self
    }

    /// Insert a middleware after a specific named middleware.
    ///
    /// If the target middleware is not found, logs a warning and does nothing.
    ///
    /// # Arguments
    ///
    /// * `target_name` - Name of the middleware to insert after
    /// * `name` - Name for the new middleware
    /// * `middleware` - The middleware implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.insert_after("logging", "metrics", Arc::new(MetricsMiddleware));
    /// ```
    pub fn insert_after(
        &mut self,
        target_name: &str,
        name: impl Into<String>,
        middleware: Arc<dyn LanguageModelMiddleware>,
    ) -> &mut Self {
        if let Some(index) = self.middlewares.iter().position(|m| m.name == target_name) {
            self.middlewares
                .insert(index + 1, NamedMiddleware::new(name, middleware));
        } else {
            tracing::warn!(
                "MiddlewareBuilder: Middleware named '{}' not found, cannot insert after",
                target_name
            );
        }
        self
    }

    /// Check if a middleware with the given name exists.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if builder.has("logging") {
    ///     println!("Logging middleware is present");
    /// }
    /// ```
    pub fn has(&self, name: &str) -> bool {
        self.middlewares.iter().any(|m| m.name == name)
    }

    /// Remove a middleware by name.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.remove("caching");
    /// ```
    pub fn remove(&mut self, name: &str) -> &mut Self {
        self.middlewares.retain(|m| m.name != name);
        self
    }

    /// Replace a middleware with a new implementation.
    ///
    /// If the middleware is not found, logs a warning and does nothing.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the middleware to replace
    /// * `middleware` - The new middleware implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.replace("logging", Arc::new(NewLoggingMiddleware));
    /// ```
    pub fn replace(
        &mut self,
        name: &str,
        middleware: Arc<dyn LanguageModelMiddleware>,
    ) -> &mut Self {
        if let Some(m) = self.middlewares.iter_mut().find(|m| m.name == name) {
            m.middleware = middleware;
        } else {
            tracing::warn!(
                "MiddlewareBuilder: Middleware named '{}' not found, cannot replace",
                name
            );
        }
        self
    }

    /// Clear all middlewares.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// builder.clear();
    /// ```
    pub fn clear(&mut self) -> &mut Self {
        self.middlewares.clear();
        self
    }

    /// Get the number of middlewares in the chain.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let count = builder.len();
    /// ```
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Check if the builder is empty.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if builder.is_empty() {
    ///     println!("No middlewares configured");
    /// }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Build the final middleware array.
    ///
    /// Consumes the builder and returns a vector of middleware implementations.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let middlewares = builder.build();
    /// ```
    pub fn build(self) -> Vec<Arc<dyn LanguageModelMiddleware>> {
        self.middlewares.into_iter().map(|m| m.middleware).collect()
    }

    /// Build and return the named middleware array (for debugging).
    ///
    /// Returns a clone of the named middlewares without consuming the builder.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let named = builder.build_named();
    /// for m in named {
    ///     println!("Middleware: {}", m.name);
    /// }
    /// ```
    pub fn build_named(&self) -> Vec<NamedMiddleware> {
        self.middlewares.clone()
    }

    /// Get a reference to the middlewares (for inspection).
    pub fn middlewares(&self) -> &[NamedMiddleware] {
        &self.middlewares
    }
}

impl std::fmt::Debug for MiddlewareBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiddlewareBuilder")
            .field("count", &self.middlewares.len())
            .field(
                "names",
                &self
                    .middlewares
                    .iter()
                    .map(|m| m.name.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}
