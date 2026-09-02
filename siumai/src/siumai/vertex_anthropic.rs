use std::fmt;

use siumai_provider_google_vertex::{
    GoogleVertexAnthropicConfigError, GoogleVertexAnthropicProvider,
    GoogleVertexAnthropicProviderBuilder, GoogleVertexCredential,
};

use super::{Siumai, SiumaiBuilder};

/// The Vertex Anthropic stage that requires a project before location and credentials.
///
/// ```compile_fail
/// use siumai::Siumai;
/// use siumai::providers::google_vertex_anthropic::GoogleVertexCredential;
///
/// let _ = Siumai::builder()
///     .vertex_anthropic()
///     .credential(GoogleVertexCredential::access_token("token"));
/// ```
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder().vertex_anthropic().build();
/// ```
#[must_use = "supply a Google Cloud project before selecting a location"]
pub struct VertexAnthropicProjectStage {
    _private: (),
}

impl VertexAnthropicProjectStage {
    const fn new() -> Self {
        Self { _private: () }
    }

    /// Select the Google Cloud project used for Vertex technical addressing.
    pub fn project(self, project: impl Into<String>) -> VertexAnthropicLocationStage {
        VertexAnthropicLocationStage {
            project: project.into(),
        }
    }
}

impl fmt::Debug for VertexAnthropicProjectStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("VertexAnthropicProjectStage")
    }
}

/// The Vertex Anthropic stage that requires a location after the project.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder()
///     .vertex_anthropic()
///     .project("project")
///     .build();
/// ```
#[must_use = "supply a Vertex location before selecting credentials"]
pub struct VertexAnthropicLocationStage {
    project: String,
}

impl VertexAnthropicLocationStage {
    /// Select the Vertex location used for technical endpoint addressing.
    pub fn location(self, location: impl Into<String>) -> VertexAnthropicCredentialStage {
        VertexAnthropicCredentialStage {
            project: self.project,
            location: location.into(),
        }
    }
}

impl fmt::Debug for VertexAnthropicLocationStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VertexAnthropicLocationStage")
            .field("project", &"configured")
            .finish_non_exhaustive()
    }
}

/// The Vertex Anthropic stage that requires a Google credential or access token.
///
/// ```compile_fail
/// use siumai::Siumai;
///
/// let _ = Siumai::builder()
///     .vertex_anthropic()
///     .project("project")
///     .location("global")
///     .build();
/// ```
#[must_use = "supply a Google Vertex credential to continue provider construction"]
pub struct VertexAnthropicCredentialStage {
    project: String,
    location: String,
}

impl VertexAnthropicCredentialStage {
    /// Use a short-lived access token and enter the buildable provider stage.
    pub fn access_token(self, token: impl Into<String>) -> VertexAnthropicProviderStage {
        self.credential(GoogleVertexCredential::access_token(token))
    }

    /// Use a complete provider-owned Google credential and enter the buildable stage.
    pub fn credential(self, credential: GoogleVertexCredential) -> VertexAnthropicProviderStage {
        VertexAnthropicProviderStage::new(GoogleVertexAnthropicProvider::builder(
            self.project,
            self.location,
            credential,
        ))
    }
}

impl fmt::Debug for VertexAnthropicCredentialStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VertexAnthropicCredentialStage")
            .field("project", &"configured")
            .field("location", &"configured")
            .finish_non_exhaustive()
    }
}

/// A buildable Vertex Anthropic stage wrapping the real provider builder.
#[must_use = "build the Vertex Anthropic provider or continue configuring it"]
pub struct VertexAnthropicProviderStage {
    builder: GoogleVertexAnthropicProviderBuilder,
}

impl VertexAnthropicProviderStage {
    fn new(builder: GoogleVertexAnthropicProviderBuilder) -> Self {
        Self { builder }
    }

    /// Configure the real Vertex Anthropic builder without copying provider-owned setters.
    pub fn configure_provider(
        self,
        configure: impl FnOnce(
            GoogleVertexAnthropicProviderBuilder,
        ) -> GoogleVertexAnthropicProviderBuilder,
    ) -> Self {
        Self::new(configure(self.builder))
    }

    /// Validate static configuration and construct one reusable typed hub.
    pub fn build(
        self,
    ) -> Result<Siumai<GoogleVertexAnthropicProvider>, GoogleVertexAnthropicConfigError> {
        self.builder.build().map(Siumai::from_provider)
    }
}

impl fmt::Debug for VertexAnthropicProviderStage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VertexAnthropicProviderStage")
            .field("provider_builder", &"configured")
            .finish_non_exhaustive()
    }
}

impl SiumaiBuilder {
    /// Select Anthropic on Google Vertex and enter its project-required stage.
    ///
    /// ```compile_fail
    /// use siumai::Siumai;
    ///
    /// let hub = Siumai::builder()
    ///     .vertex_anthropic()
    ///     .project("project")
    ///     .location("global")
    ///     .access_token("token")
    ///     .build()
    ///     .unwrap();
    /// let _ = hub.embedding("unsupported-model");
    /// ```
    pub const fn vertex_anthropic(self) -> VertexAnthropicProjectStage {
        VertexAnthropicProjectStage::new()
    }
}
