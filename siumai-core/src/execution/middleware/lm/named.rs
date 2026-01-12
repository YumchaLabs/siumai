//! Named middleware for identification and manipulation.
//! (moved under execution/middleware/lm)

use super::LanguageModelMiddleware;
use std::sync::Arc;

/// Named middleware wrapper that associates a name with a middleware instance.
///
/// This allows middleware to be identified, queried, removed, or replaced by name,
/// providing a more flexible middleware management system.
///
/// # Example
///
/// ```rust,ignore
/// use siumai::experimental::execution::middleware::{NamedMiddleware, LanguageModelMiddleware};
/// use std::sync::Arc;
///
/// let middleware = Arc::new(MyMiddleware::new());
/// let named = NamedMiddleware::new("my-middleware", middleware);
/// ```
#[derive(Clone)]
pub struct NamedMiddleware {
    /// Unique name for this middleware
    pub name: String,
    /// The actual middleware implementation
    pub middleware: Arc<dyn LanguageModelMiddleware>,
}

impl NamedMiddleware {
    /// Create a new named middleware.
    ///
    /// # Arguments
    ///
    /// * `name` - A unique identifier for this middleware
    /// * `middleware` - The middleware implementation
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let named = NamedMiddleware::new("logging", Arc::new(LoggingMiddleware));
    /// ```
    pub fn new(name: impl Into<String>, middleware: Arc<dyn LanguageModelMiddleware>) -> Self {
        Self {
            name: name.into(),
            middleware,
        }
    }
}

impl std::fmt::Debug for NamedMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedMiddleware")
            .field("name", &self.name)
            .field("middleware", &"<dyn LanguageModelMiddleware>")
            .finish()
    }
}
